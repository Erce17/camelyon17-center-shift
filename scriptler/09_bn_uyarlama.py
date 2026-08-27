"""
8. BLOK: ETIKETSIZ UYARLAMA (BatchNorm).

Blok 7 sonucu: dis merkezdeki dogruluk kaybinin yarisi SADECE esik tasiyarak geri
geliyor. Kalan yarisi siralamanin kendisinde kaybolmus ve esik ayariyla gelmiyor.

Bu script o kalan yariya dokunmayi dener, ve ETIKET KULLANMADAN.

Fikir: ResNet-18'in icinde 20 BatchNorm katmani var. Her biri kendinden onceki
katmanin ciktisini normalize eder, ve kullandigi ortalama/varyans EGITIM
VERISININ istatistikleridir. Model dis merkeze goturuldugunde iceride hala egitim
hastanelerinin dagilimina gore normalize eder: yanlis zemin, ve hata her katmanda
buyuyerek ilerler.

Cozum: hedef merkezin ETIKETSIZ karolarini modelden bir kez gecir, BN'lerin
biriktirdigi istatistikleri o merkeze gore yeniden hesapla. AGIRLIKLARA DOKUNMA.
Ortalama ve varyans hesaplamak icin etikete ihtiyac yok -- sadece goruntuye.

Neden bu veride isleme sansi yuksek: WILDS'ta tumor orani bes merkezde de %50.
Yani SAF KOVARYAT KAYMASI var, etiket kaymasi yok. BN uyarlamasi tam bu duruma
gore tasarlanmis bir sey.

AUC bu blokta RAPORLANIR: esik ayari siralamayi degistiremezdi, BN uyarlamasi
degistirebilir. Ilk kez agin cikisina degil ICINE dokunuyoruz.

Kollar:
  taban        : uyarlamasiz, esik ic dogrulamadan (sahadaki varsayilan)
  esik200      : esik dis merkezden 200 ETIKETLI ornekle (Blok 7'nin en iyisi)
  bn100        : BN uyarlamasi, 100 ETIKETSIZ karo
  bn_tam       : BN uyarlamasi, havuzun tamami (etiketsiz)
  bn_tam+esik  : ikisi birden

SIZINTI KURALI: uyarlama ve kalibrasyon karolari, olcum yapilan hastalardan olamaz.
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from torchvision.models import resnet18, ResNet18_Weights

TOHUMLAR = [42, 43, 44]
MERKEZLER = [0, 1, 2, 3, 4]
MERKEZ_BASI = 10_000        # egitim butcesi, her katta sabit
IC_DOGRULAMA_HASTA = 1      # her egitim merkezinden ayrilir -> 3 hasta
OLCUM_HASTA = 3             # her olcum kumesinde hasta sayisi (esitleme)
KALIBRASYON_N = 200          # Blok 7'nin en iyi noktasi
BN_KARO = 100                # etiketsiz uyarlama icin kucuk kume
HAVUZ_TAVAN = 800            # kalibrasyon/uyarlama havuzunun ust siniri
HASTA_BASI_SINIF = 150      # hasta basina her siniftan karo -> hasta basi 300
TUR = 5
YIGIN = 128

aygit = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"aygit: {aygit}", flush=True)

X = np.load("veri/altkume/X.npy", mmap_mode="r")
md = pd.read_parquet("veri/altkume/secim.parquet").reset_index(drop=True)
md["idx"] = np.arange(len(md))
etiketler = md.label.values.astype(np.float32)

# ---- OLCUME UYGUN HASTALAR ------------------------------------------------
# 50/50 denge veri setinin KURESEL ozelligi; hasta duzeyinde cogu hastada tumor
# dokusu yok denecek kadar az. Merkez basina her siniftan >=150 karosu olan hasta
# sayisi: 5, 5, 5, 3, 3. Olcum kumesinin gercek birim sayisi bu -- 30 bin karo
# degil. Kume boyutu bu tavana gore kuruluyor, tersi degil.
_h = md.groupby(["center", "patient"]).label.agg(["size", "sum"])
_h["min_sinif"] = np.minimum(_h["sum"], _h["size"] - _h["sum"])
UYGUN = {m: np.sort(d[d.min_sinif >= HASTA_BASI_SINIF].index.get_level_values(1).values)
         for m, d in _h.groupby(level=0)}
print("olcume uygun hasta sayisi (merkez basi): "
      + str({int(m): len(v) for m, v in UYGUN.items()}), flush=True)


def olcum_kumesi(kaynak, hastalar, tohum):
    """Hasta basina esit, hasta icinde 50/50. Yetersiz hasta atlanir."""
    parcalar, kullanilan = [], []
    for h in hastalar:
        g = kaynak[kaynak.patient == h]
        t, n = g[g.label == 1], g[g.label == 0]
        k = min(HASTA_BASI_SINIF, len(t), len(n))
        if k == 0:
            continue
        parcalar.append(t.sample(n=k, random_state=tohum))
        parcalar.append(n.sample(n=k, random_state=tohum))
        kullanilan.append((int(h), k * 2))
    return pd.concat(parcalar), kullanilan


def yigin_getir(idxler, karistir, artir, ort_t, std_t):
    sira = np.random.permutation(len(idxler)) if karistir else np.arange(len(idxler))
    for i in range(0, len(sira), YIGIN):
        alt = np.sort(idxler[sira[i:i + YIGIN]])
        x = torch.from_numpy(np.asarray(X[alt]).copy()).to(aygit)
        x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        if artir:
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        yield (x - ort_t) / std_t, alt


def bn_uyarla(model, idxler, ort_t, std_t):
    """Hedef merkezin ETIKETSIZ karolariyla BN istatistiklerini yeniden hesaplar.
    Agirliklar degismez: opt.step() yok, gradyan yok."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None          # kumulatif ortalama, tek gecis yeter
    model.train()
    with torch.no_grad():
        for x, _ in yigin_getir(idxler, False, False, ort_t, std_t):
            model(x)                   # etiket hic kullanilmiyor
    model.eval()
    return model


def esik_bul(skor, y, ozgulluk_hedef=0.90):
    """Verilen kumede istenen ozgullugu saglayan en dusuk esik."""
    negatif = np.sort(skor[y == 0])
    if len(negatif) == 0:
        return 0.5
    return float(negatif[min(int(len(negatif) * ozgulluk_hedef), len(negatif) - 1)])


def olc(skor, y, esik=0.5):
    fpr, tpr, _ = roc_curve(y, skor)
    return dict(auc=roc_auc_score(y, skor),
                dogruluk=float(((skor >= esik).astype(int) == y).mean()),
                duyarlilik_ozg90=float(np.interp(0.10, fpr, tpr)),
                brier=brier_score_loss(y, skor))


def kosu(kat, tohum):
    dis_test = kat
    dis_dog = (kat + 1) % 5
    egitim_merkez = [m for m in MERKEZLER if m not in (dis_test, dis_dog)]
    torch.manual_seed(tohum); np.random.seed(tohum)
    rng = np.random.default_rng(tohum)

    # --- hasta bazli ayrim: ic dogrulama hastalari egitimde GORULMEZ ---
    ic_hastalar = []
    for m in egitim_merkez:
        ic_hastalar += list(rng.choice(UYGUN[m], IC_DOGRULAMA_HASTA, replace=False))

    ic_kaynak = md[md.center.isin(egitim_merkez) & md.patient.isin(ic_hastalar)]
    ic_dog, ic_kul = olcum_kumesi(ic_kaynak, ic_hastalar, tohum)

    # --- dis kumeler: AYNI hasta sayisina indirilir ---
    dis = {}
    olcum_hastalari = {}
    for ad, m in [("dis_dogrulama", dis_dog), ("dis_test", dis_test)]:
        kaynak = md[md.center == m]
        sec = rng.choice(UYGUN[m], min(OLCUM_HASTA, len(UYGUN[m])), replace=False)
        dis[ad] = olcum_kumesi(kaynak, sec, tohum)
        olcum_hastalari[m] = set(int(h) for h in sec)

    # --- KALIBRASYON HAVUZU: dis test merkezinin olcumde KULLANILMAYAN hastalari ---
    kaynak_dt = md[md.center == dis_test]
    kalib_hastalar = [h for h in np.sort(kaynak_dt.patient.unique())
                      if int(h) not in olcum_hastalari[dis_test]]
    kh = kaynak_dt[kaynak_dt.patient.isin(kalib_hastalar)]
    _t, _n = kh[kh.label == 1], kh[kh.label == 0]
    _k = min(len(_t), len(_n), HAVUZ_TAVAN // 2)
    kalib_havuz = (pd.concat([_t.sample(n=_k, random_state=tohum),
                              _n.sample(n=_k, random_state=tohum)])
                   if _k > 0 else kh.iloc[0:0])

    # --- egitim: merkez basina esit, etiket dengeli ---
    havuz = md[md.center.isin(egitim_merkez) & ~md.patient.isin(ic_hastalar)]
    parcalar = []
    for m in egitim_merkez:
        g = havuz[havuz.center == m]
        for e in (0, 1):
            alt = g[g.label == e]
            parcalar.append(alt.sample(n=min(MERKEZ_BASI // 2, len(alt)),
                                       random_state=tohum))
    egitim = pd.concat(parcalar)

    print(f"\n{'='*74}\nKAT {kat} (dis test m{dis_test}, dis dog m{dis_dog}, "
          f"egitim m{egitim_merkez}) | TOHUM {tohum}\n{'='*74}", flush=True)
    print(f"{'egitim':>22}: {len(egitim):>6,} karo | {egitim.patient.nunique():>2} hasta", flush=True)
    print(f"{'kalibrasyon havuzu':>22}: {len(kalib_havuz):>6,} karo | "
          f"{kalib_havuz.patient.nunique():>2} hasta (olcumde KULLANILMAYAN)", flush=True)
    for ad, (d, kul) in [("ic dogrulama", (ic_dog, ic_kul)),
                         ("dis dogrulama", dis["dis_dogrulama"]),
                         ("dis test", dis["dis_test"])]:
        print(f"{ad:>22}: {len(d):>6,} karo | {d.patient.nunique():>2} hasta | "
              f"tumor %{d.label.mean()*100:.1f} | hasta basi {sorted(k for _, k in kul)}", flush=True)

    # --- normalizasyon: SADECE egitim karolarindan ---
    ornek = np.asarray(X[np.sort(egitim.idx.values)[:5000]]).astype(np.float32) / 255.0
    ort_t = torch.tensor(ornek.mean(axis=(0, 1, 2)), device=aygit).view(1, 3, 1, 1)
    std_t = torch.tensor(ornek.std(axis=(0, 1, 2)), device=aygit).view(1, 3, 1, 1)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 1)
    model = model.to(aygit)
    kayip_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    def skorla_ile(model_, d):
        model_.eval(); skor = np.zeros(len(d), dtype=np.float32)
        yer = {v: i for i, v in enumerate(d.idx.values)}
        with torch.no_grad():
            for x, alt in yigin_getir(d.idx.values, False, False, ort_t, std_t):
                p = torch.sigmoid(model_(x)).squeeze(1).cpu().numpy()
                for j, a in enumerate(alt): skor[yer[a]] = p[j]
        return skor, d.label.values

    en_iyi, en_iyi_durum = -1, None
    egitim_idx = egitim.idx.values
    for tur in range(1, TUR + 1):
        model.train(); toplam, n = 0.0, 0
        for x, alt in yigin_getir(egitim_idx, True, True, ort_t, std_t):
            y = torch.from_numpy(etiketler[alt]).to(aygit)
            opt.zero_grad()
            kayip = kayip_fn(model(x).squeeze(1), y)
            kayip.backward(); opt.step()
            toplam += kayip.item() * len(alt); n += len(alt)
        s, y = skorla_ile(model, ic_dog); m = olc(s, y)
        print(f"  tur {tur}: kayip {toplam/n:.4f} | ic AUC {m['auc']:.4f} "
              f"dogruluk {m['dogruluk']:.4f}", flush=True)
        if m["auc"] > en_iyi:   # model secimi IC dogrulamada
            en_iyi = m["auc"]
            en_iyi_durum = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(en_iyi_durum)
    temiz_durum = {k: v.detach().clone() for k, v in model.state_dict().items()}
    dt = dis["dis_test"][0]

    def deger(ad, model_):
        s_ic, y_ic = skorla_ile(model_, ic_dog)
        s_dt, y_dt = skorla_ile(model_, dt)
        e = esik_bul(s_ic, y_ic)                     # esik IC dogrulamadan
        m = olc(s_dt, y_dt, e)
        return dict(kat=kat, tohum=tohum, dis_test_merkez=dis_test, kol=ad,
                    ic_auc=roc_auc_score(y_ic, s_ic), **m), (s_dt, y_dt)

    satirlar = []
    r, (s_dt, y_dt) = deger("taban", model); satirlar.append(r)
    taban_dog = r["dogruluk"]

    # --- esik200: dis merkezden ETIKETLI 200 ornek (Blok 7) ---
    if len(kalib_havuz) > 0:
        s_kal, y_kal = skorla_ile(model, kalib_havuz)
        n_al = min(KALIBRASYON_N // 2, len(s_kal) // 2)
        poz = np.where(y_kal == 1)[0][:n_al]; neg = np.where(y_kal == 0)[0][:n_al]
        sec = np.concatenate([poz, neg])
        e_dis = esik_bul(s_kal[sec], y_kal[sec])
        m = olc(s_dt, y_dt, e_dis)
        satirlar.append(dict(kat=kat, tohum=tohum, dis_test_merkez=dis_test,
                             kol="esik200", ic_auc=np.nan, **m))

    # --- BN uyarlamasi: ETIKETSIZ ---
    for ad, n_karo in [("bn100", BN_KARO), ("bn_tam", len(kalib_havuz))]:
        if len(kalib_havuz) < 10:
            continue
        model.load_state_dict(temiz_durum)
        idx_uy = kalib_havuz.idx.values[:n_karo]
        bn_uyarla(model, idx_uy, ort_t, std_t)
        r2, (s2, y2) = deger(ad, model)
        satirlar.append(r2)
        if ad == "bn_tam":
            s_k2, y_k2 = skorla_ile(model, kalib_havuz)
            e2 = esik_bul(s_k2[sec], y_k2[sec]) if len(kalib_havuz) > 0 else 0.5
            m2 = olc(s2, y2, e2)
            satirlar.append(dict(kat=kat, tohum=tohum, dis_test_merkez=dis_test,
                                 kol="bn_tam+esik", ic_auc=np.nan, **m2))
    model.load_state_dict(temiz_durum)

    df = pd.DataFrame(satirlar)
    ozet = " | ".join(f"{r.kol} {r.dogruluk:.4f}" for r in df.itertuples())
    print(f"  -> {ozet}", flush=True)
    return df


t0 = time.time()
hepsi = []
for kat in MERKEZLER:
    for tohum in TOHUMLAR:
        ts = time.time()
        hepsi.append(kosu(kat, tohum))
        print(f"  [kosu {len(hepsi)}/15 bitti, {time.time()-ts:.0f} sn, "
              f"toplam {(time.time()-t0)/60:.1f} dk]", flush=True)
        pd.concat(hepsi).to_csv("sonuclar/sonuc_bn.csv", index=False)

son = pd.concat(hepsi)
son.to_csv("sonuclar/sonuc_bn.csv", index=False)
print(f"\n{'='*84}\nOZET ({(time.time()-t0)/60:.1f} dakika)\n{'='*84}", flush=True)

tavan = son[son.kol == "taban"].ic_auc.mean()
tb = son[son.kol == "taban"]
print(f"\nic dogrulama AUC (tavan referansi): {tavan:.4f}\n")
print(f"{'kol':>13} {'dis AUC':>9} {'dogruluk':>9} {'duy@ozg90':>10} {'brier':>8}  {'etiket?':>8}")
etiket = {"taban": "-", "esik200": "200 adet", "bn100": "YOK", "bn_tam": "YOK",
          "bn_tam+esik": "200 adet"}
for kol in ["taban", "esik200", "bn100", "bn_tam", "bn_tam+esik"]:
    g = son[son.kol == kol]
    if len(g) == 0:
        continue
    print(f"{kol:>13} {g.auc.mean():>9.4f} {g.dogruluk.mean():>9.4f} "
          f"{g.duyarlilik_ozg90.mean():>10.4f} {g.brier.mean():>8.4f}  {etiket[kol]:>8}")

t_auc = tb.auc.mean(); t_dog = tb.dogruluk.mean()
print(f"\n{'kol':>13} {'AUC degisim':>12} {'dogruluk degisim':>17}")
for kol in ["esik200", "bn100", "bn_tam", "bn_tam+esik"]:
    g = son[son.kol == kol]
    if len(g) == 0:
        continue
    print(f"{kol:>13} {g.auc.mean()-t_auc:>+12.4f} {g.dogruluk.mean()-t_dog:>+17.4f}")

print(f"\nkat basina dis AUC:")
p = son.pivot_table(index="kat", columns="kol", values="auc")
print(p.round(4).to_string())
print("\nsonuclar/sonuc_bn.csv yazildi", flush=True)
