"""
7. BLOK: YENIDEN KALIBRASYON.

Blok 6 ve teshis ikinci turu su tabloyu birakti: dis merkezde AUC %2,5 kaybediliyor
ama brier %43 bozuluyor. SIRALAMA tasiniyor, ESIK ve KALIBRASYON tasinmiyor.

Bunun dogrudan ve test edilebilir bir sonucu var:
  Hedef hastaneden AZ SAYIDA etiketli ornek alip sadece esigi/kalibrasyonu yeniden
  ayarlarsak kaybin ne kadari geri gelir, ve N kac olmali?

Model YENIDEN EGITILMEZ. Skorlarin uzerine bir donusum takilir, o kadar.

SIZINTI KURALI: kalibrasyon karolari degerlendirme karolariyla AYNI HASTADAN olamaz.
Kalibrasyon seti, dis merkezin olcum kumesine girmeyen hastalarindan cekilir. Aksi
halde "yeni hastaneye uyarladim" derken test hastasi ezberlenir.

Uc yontem:
  esik_ic  : ic dogrulamada ozgulluk 0,90 veren esik (sahadaki varsayilan durum)
  esik_dis : dis kalibrasyon setinden N ornekle bulunan esik
  platt    : N ornekle egitilen tek degiskenli lojistik regresyon (olasilik duzeltme)

AUC raporlanmaz: monoton donusumler AUC'yi degistirmez ve kaybin AUC'de olmadigini
zaten biliyoruz.
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
KALIBRASYON_N = [20, 50, 100, 200, 400]   # dis merkezden alinan etiketli karo sayisi
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
    _k = min(len(_t), len(_n), max(KALIBRASYON_N))
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

    def skorla(d):
        model.eval(); skor = np.zeros(len(d), dtype=np.float32)
        yer = {v: i for i, v in enumerate(d.idx.values)}
        with torch.no_grad():
            for x, alt in yigin_getir(d.idx.values, False, False, ort_t, std_t):
                p = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
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
        s, y = skorla(ic_dog); m = olc(s, y)
        print(f"  tur {tur}: kayip {toplam/n:.4f} | ic AUC {m['auc']:.4f} "
              f"dogruluk {m['dogruluk']:.4f}", flush=True)
        if m["auc"] > en_iyi:   # model secimi IC dogrulamada
            en_iyi = m["auc"]
            en_iyi_durum = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(en_iyi_durum)
    s_ic, y_ic = skorla(ic_dog)
    s_dt, y_dt = skorla(dis["dis_test"][0])

    # sahadaki varsayilan: esik IC dogrulamada secilir
    esik_ic = esik_bul(s_ic, y_ic)
    ic_ol = olc(s_ic, y_ic, esik_ic)
    tavan = ic_ol["dogruluk"]                       # ulasilabilecek en iyi (ic)
    taban = olc(s_dt, y_dt, esik_ic)["dogruluk"]    # mudahalesiz dis
    taban_brier = brier_score_loss(y_dt, s_dt)

    satirlar = [dict(kat=kat, tohum=tohum, dis_test_merkez=dis_test, yontem="ic_tavan",
                     n=0, dogruluk=tavan, brier=brier_score_loss(y_ic, s_ic),
                     duyarlilik=olc(s_ic, y_ic, esik_ic)["duyarlilik_ozg90"]),
                dict(kat=kat, tohum=tohum, dis_test_merkez=dis_test, yontem="esik_ic",
                     n=0, dogruluk=taban, brier=taban_brier,
                     duyarlilik=olc(s_dt, y_dt, esik_ic)["duyarlilik_ozg90"])]

    if len(kalib_havuz) > 0:
        s_kal_tum, y_kal_tum = skorla(kalib_havuz)
        for N in KALIBRASYON_N:
            n_al = min(N // 2, len(s_kal_tum) // 2)
            if n_al < 2:
                continue
            poz = np.where(y_kal_tum == 1)[0][:n_al]
            neg = np.where(y_kal_tum == 0)[0][:n_al]
            sec = np.concatenate([poz, neg])
            sk, yk = s_kal_tum[sec], y_kal_tum[sec]

            e_dis = esik_bul(sk, yk)
            satirlar.append(dict(kat=kat, tohum=tohum, dis_test_merkez=dis_test,
                                 yontem="esik_dis", n=len(sec),
                                 dogruluk=olc(s_dt, y_dt, e_dis)["dogruluk"],
                                 brier=taban_brier,
                                 duyarlilik=olc(s_dt, y_dt, e_dis)["duyarlilik_ozg90"]))

            if len(np.unique(yk)) == 2:
                pl = LogisticRegression(max_iter=1000).fit(sk.reshape(-1, 1), yk)
                s_pl = pl.predict_proba(s_dt.reshape(-1, 1))[:, 1]
                e_pl = esik_bul(pl.predict_proba(sk.reshape(-1, 1))[:, 1], yk)
                satirlar.append(dict(kat=kat, tohum=tohum, dis_test_merkez=dis_test,
                                     yontem="platt", n=len(sec),
                                     dogruluk=olc(s_pl, y_dt, e_pl)["dogruluk"],
                                     brier=brier_score_loss(y_dt, s_pl),
                                     duyarlilik=olc(s_pl, y_dt, e_pl)["duyarlilik_ozg90"]))

    df = pd.DataFrame(satirlar)
    en_iyi_n = df[df.yontem == "platt"].dogruluk.max() if (df.yontem == "platt").any() else np.nan
    print(f"  -> ic tavan {tavan:.4f} | dis taban {taban:.4f} | "
          f"platt en iyi {en_iyi_n:.4f} | acik {tavan - taban:+.4f}", flush=True)
    return df


t0 = time.time()
hepsi = []
for kat in MERKEZLER:
    for tohum in TOHUMLAR:
        ts = time.time()
        hepsi.append(kosu(kat, tohum))
        print(f"  [kosu {len(hepsi)}/15 bitti, {time.time()-ts:.0f} sn, "
              f"toplam {(time.time()-t0)/60:.1f} dk]", flush=True)
        pd.concat(hepsi).to_csv("sonuclar/sonuc_tam.csv", index=False)

son = pd.concat(hepsi)
son.to_csv("sonuclar/sonuc_kalibrasyon.csv", index=False)
print(f"\n{'='*74}\nOZET ({(time.time()-t0)/60:.1f} dakika)\n{'='*74}", flush=True)

tavan = son[son.yontem == "ic_tavan"].dogruluk.mean()
taban = son[son.yontem == "esik_ic"].dogruluk.mean()
acik = tavan - taban
print(f"\nic tavan (dogruluk)      : {tavan:.4f}")
print(f"dis taban (mudahalesiz)  : {taban:.4f}")
print(f"KAPATILACAK ACIK         : {acik:+.4f}\n")

print(f"{'yontem':>10} {'N':>5} {'dogruluk':>9} {'kapanan acik':>13} {'brier':>8} {'duyarlilik':>11}")
for yontem in ["esik_dis", "platt"]:
    for N in KALIBRASYON_N:
        g = son[(son.yontem == yontem) & (son.n == N)]
        if len(g) == 0:
            continue
        d = g.dogruluk.mean()
        pay = (d - taban) / acik * 100 if abs(acik) > 1e-9 else float("nan")
        print(f"{yontem:>10} {N:>5} {d:>9.4f} {pay:>12.1f}% {g.brier.mean():>8.4f} "
              f"{g.duyarlilik.mean():>11.4f}")

print(f"\nreferans: mudahalesiz brier {son[son.yontem=='esik_ic'].brier.mean():.4f} "
      f"| ic brier {son[son.yontem=='ic_tavan'].brier.mean():.4f}")
print("\nsonuclar/sonuc_kalibrasyon.csv yazildi", flush=True)
