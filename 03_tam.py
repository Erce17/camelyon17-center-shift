"""
3. ADIM: TAM KOSU. Pilotun uc hatasini duzeltir.

1. OLCUM KUMELERI ESITLENDI. Pilotta ic dogrulama 6 hastadan 2.612 karo,
   dis kumeler 8-9 hastadan 30.000 karoydu. Merkez kaymasiyla hasta kaymasi
   kiyaslanmisti. Simdi ucu de: ayni hasta sayisi, hasta basina ayni karo,
   hasta icinde 50/50 etiket. Tek hastanin kumeyi domine etmesi de boylece biter.

2. BES KAT. Her merkez sirayla disarida. kat k: dis test = k,
   dis dogrulama = (k+1)%5, egitim = kalan uc merkez. Tek merkezin
   tesadufu sonucu belirlemesin.

3. UC TOHUM. Camelyon17'de dis basarim tohumdan tohuma oynar. Tek sayi degil
   aralik raporlanir. Tohum hem hasta secimini hem model baslangicini etkiler.

Degismeyenler: model secimi IC dogrulamada, normalizasyon yalniz egitimden,
taban cizgisinde renk artirma yok, her katta ayni egitim butcesi.
"""
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from torchvision.models import resnet18, ResNet18_Weights

TOHUMLAR = [42, 43, 44]
MERKEZLER = [0, 1, 2, 3, 4]
MERKEZ_BASI = 10_000        # egitim butcesi, her katta sabit
IC_DOGRULAMA_HASTA = 1      # her egitim merkezinden ayrilir -> 3 hasta
OLCUM_HASTA = 3             # her olcum kumesinde hasta sayisi (esitleme)
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
    for ad, m in [("dis_dogrulama", dis_dog), ("dis_test", dis_test)]:
        kaynak = md[md.center == m]
        sec = rng.choice(UYGUN[m], min(OLCUM_HASTA, len(UYGUN[m])), replace=False)
        dis[ad] = olcum_kumesi(kaynak, sec, tohum)

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
    satirlar = []
    for ad, d in [("ic_dogrulama", ic_dog),
                  ("dis_dogrulama", dis["dis_dogrulama"][0]),
                  ("dis_test", dis["dis_test"][0])]:
        s, y = skorla(d)
        satirlar.append(dict(kat=kat, tohum=tohum, dis_test_merkez=dis_test,
                             kume=ad, n_karo=len(d), n_hasta=int(d.patient.nunique()),
                             **olc(s, y)))
    df = pd.DataFrame(satirlar)
    ic_auc = df[df.kume == "ic_dogrulama"].auc.iloc[0]
    dt_auc = df[df.kume == "dis_test"].auc.iloc[0]
    print(f"  -> ic {ic_auc:.4f} | dis test {dt_auc:.4f} | "
          f"DUSUS {ic_auc - dt_auc:+.4f} AUC", flush=True)
    return df


t0 = time.time()
hepsi = []
for kat in MERKEZLER:
    for tohum in TOHUMLAR:
        ts = time.time()
        hepsi.append(kosu(kat, tohum))
        print(f"  [kosu {len(hepsi)}/15 bitti, {time.time()-ts:.0f} sn, "
              f"toplam {(time.time()-t0)/60:.1f} dk]", flush=True)
        pd.concat(hepsi).to_csv("sonuc_tam.csv", index=False)

son = pd.concat(hepsi)
print(f"\n{'='*74}\nOZET ({(time.time()-t0)/60:.1f} dakika)\n{'='*74}", flush=True)
print("\nKume basina, 15 kosu (ortalama +- std, [min, max]):")
for k in ["ic_dogrulama", "dis_dogrulama", "dis_test"]:
    g = son[son.kume == k]
    print(f"{k:>16}: AUC {g.auc.mean():.4f} +- {g.auc.std():.4f} "
          f"[{g.auc.min():.4f}, {g.auc.max():.4f}] | "
          f"dogruluk {g.dogruluk.mean():.4f} +- {g.dogruluk.std():.4f} | "
          f"brier {g.brier.mean():.4f}", flush=True)

print("\nKat basina dusus (ic - dis test, uc tohum):")
p = son.pivot_table(index=["kat", "tohum"], columns="kume", values="auc")
p["dusus"] = p.ic_dogrulama - p.dis_test
for kat, g in p.groupby("kat"):
    print(f"  kat {kat} (dis test m{kat}): dusus {g.dusus.mean():+.4f} "
          f"[{g.dusus.min():+.4f}, {g.dusus.max():+.4f}]", flush=True)
print(f"\nGENEL DUSUS: {p.dusus.mean():+.4f} +- {p.dusus.std():.4f} AUC", flush=True)
print("\nsonuc_tam.csv yazildi", flush=True)
