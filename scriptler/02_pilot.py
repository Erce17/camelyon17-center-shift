"""
2. ADIM: PILOT. Tek kat, kucuk butce. Amac sonuc almak degil, HATTIN CALISTIGINI
gormek: veri okunuyor mu, MPS'te egitim donuyor mu, ic ve dis olcum cikiyor mu.

Bolunme WILDS'in kendi bolunmesi:
    egitim      : merkez 0, 3, 4  (hastalarin bir kismi ic dogrulamaya ayrilir)
    ic dogrulama: ayni merkezler, EGITIMDE GORULMEYEN hastalar
    dis dogrulama: merkez 1   -> egitim boyunca hic bakilmaz
    dis test     : merkez 2   -> egitim boyunca hic bakilmaz

Model secimi IC dogrulamada yapilir. Dis kumede yapmak, hedef hastaneden
bilgi sizdirmaktir: sahada yeni hastanenin etiketli verisi elinde olmaz.

Taban cizgisinde renk artirma YOK. Renk artirma bizim tedavimiz; tedaviyi
kontrol grubuna koyarsak ise yarayip yaramadigini olcemeyiz.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from torchvision.models import resnet18, ResNet18_Weights

TOHUM = 42
EGITIM_MERKEZ = [0, 3, 4]
DIS_DOGRULAMA = 1
DIS_TEST = 2
MERKEZ_BASI = 10_000      # pilot butcesi
IC_DOGRULAMA_HASTA = 2    # her egitim merkezinden ayrilacak hasta sayisi
TUR = 5
YIGIN = 128

torch.manual_seed(TOHUM); np.random.seed(TOHUM)
aygit = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"aygit: {aygit}")

X = np.load("veri/altkume/X.npy", mmap_mode="r")
md = pd.read_parquet("veri/altkume/secim.parquet").reset_index(drop=True)
md["idx"] = np.arange(len(md))

# ---- BOLUNME: once hasta bazinda, sonra karo bazinda -----------------------
rng = np.random.default_rng(TOHUM)
ic_dogrulama_hastalar = []
for m in EGITIM_MERKEZ:
    hastalar = np.sort(md[md.center == m].patient.unique())
    ic_dogrulama_hastalar += list(rng.choice(hastalar, IC_DOGRULAMA_HASTA, replace=False))
print(f"ic dogrulamaya ayrilan hastalar: {sorted(int(h) for h in ic_dogrulama_hastalar)}")

egitim_havuz = md[md.center.isin(EGITIM_MERKEZ) & ~md.patient.isin(ic_dogrulama_hastalar)]
ic_dog = md[md.center.isin(EGITIM_MERKEZ) & md.patient.isin(ic_dogrulama_hastalar)]
# 50/50 denge veri setinin KURESEL ozelligi, hasta duzeyinde degil: hastalarin
# cogunda tumor dokusu az. Hasta bazli ayrim korunuyor (sizinti olmasin diye),
# ama olculer kiyaslanabilsin diye ic dogrulama 50/50ye indiriliyor.
_t = ic_dog[ic_dog.label == 1]
_n = ic_dog[ic_dog.label == 0].sample(n=len(_t), random_state=TOHUM)
ic_dog = pd.concat([_t, _n])
dis_dog = md[md.center == DIS_DOGRULAMA]
dis_test = md[md.center == DIS_TEST]

# egitim butcesi: merkez basina esit, etiket bazinda dengeli
parcalar = []
for m in EGITIM_MERKEZ:
    g = egitim_havuz[egitim_havuz.center == m]
    for e in (0, 1):
        alt = g[g.label == e]
        parcalar.append(alt.sample(n=min(MERKEZ_BASI // 2, len(alt)), random_state=TOHUM))
egitim = pd.concat(parcalar)

for ad, d in [("egitim", egitim), ("ic dogrulama", ic_dog),
              ("dis dogrulama (merkez 1)", dis_dog), ("dis test (merkez 2)", dis_test)]:
    print(f"{ad:>26}: {len(d):>6,} karo | {d.patient.nunique():>2} hasta | "
          f"tumor %{d.label.mean()*100:.1f}")

# ---- NORMALIZASYON: SADECE egitim karolarindan ----------------------------
ornek = np.asarray(X[np.sort(egitim.idx.values)[:5000]]).astype(np.float32) / 255.0
ort = ornek.mean(axis=(0, 1, 2))
std = ornek.std(axis=(0, 1, 2))
print(f"\nnormalizasyon (yalniz egitimden): ort={ort.round(3)} std={std.round(3)}")
ort_t = torch.tensor(ort, device=aygit).view(1, 3, 1, 1)
std_t = torch.tensor(std, device=aygit).view(1, 3, 1, 1)


def yigin_getir(idxler, karistir, artir):
    sira = np.random.permutation(len(idxler)) if karistir else np.arange(len(idxler))
    for i in range(0, len(sira), YIGIN):
        alt = np.sort(idxler[sira[i:i + YIGIN]])
        x = torch.from_numpy(np.asarray(X[alt]).copy()).to(aygit)
        x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)  # MPS: permute sonrasi bellek bitisik degil
        if artir:  # patolojide yon anlamli degil: iki eksende de cevirme mesru
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
            if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
        yield (x - ort_t) / std_t, alt


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 1)
model = model.to(aygit)
kayip_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

etiketler = md.label.values.astype(np.float32)


def skorla(d):
    model.eval(); skor = np.zeros(len(d), dtype=np.float32)
    yer = {v: i for i, v in enumerate(d.idx.values)}
    with torch.no_grad():
        for x, alt in yigin_getir(d.idx.values, False, False):
            p = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            for j, a in enumerate(alt): skor[yer[a]] = p[j]
    return skor, d.label.values


def olc(skor, y, esik=0.5):
    auc = roc_auc_score(y, skor)
    tah = (skor >= esik).astype(int)
    dogruluk = (tah == y).mean()
    fpr, tpr, _ = roc_curve(y, skor)
    duyarlilik_90 = float(np.interp(0.10, fpr, tpr))   # ozgulluk %90 sabitken
    return dict(auc=auc, dogruluk=dogruluk, duyarlilik_ozg90=duyarlilik_90,
                brier=brier_score_loss(y, skor))


print("\n" + "=" * 74)
print("EGITIM")
print("=" * 74)
en_iyi, en_iyi_durum = -1, None
egitim_idx = egitim.idx.values
for tur in range(1, TUR + 1):
    model.train(); toplam, n = 0.0, 0
    for x, alt in yigin_getir(egitim_idx, True, True):
        y = torch.from_numpy(etiketler[alt]).to(aygit)
        opt.zero_grad()
        kayip = kayip_fn(model(x).squeeze(1), y)
        kayip.backward(); opt.step()
        toplam += kayip.item() * len(alt); n += len(alt)
    s, y = skorla(ic_dog); m = olc(s, y)
    print(f"tur {tur}: kayip {toplam/n:.4f} | ic dogrulama AUC {m['auc']:.4f} "
          f"dogruluk {m['dogruluk']:.4f}")
    if m["auc"] > en_iyi:
        en_iyi = m["auc"]
        en_iyi_durum = {k: v.detach().clone() for k, v in model.state_dict().items()}

model.load_state_dict(en_iyi_durum)
print(f"\nsecilen model: ic dogrulama AUC {en_iyi:.4f}")

print("\n" + "=" * 74)
print("SONUCLAR")
print("=" * 74)
print(f"{'kume':>26} {'AUC':>7} {'dogruluk':>9} {'duy@ozg90':>10} {'brier':>7}")
satirlar = []
for ad, d in [("ic dogrulama (0,3,4)", ic_dog),
              ("dis dogrulama (merkez 1)", dis_dog),
              ("dis test (merkez 2)", dis_test)]:
    s, y = skorla(d); m = olc(s, y)
    print(f"{ad:>26} {m['auc']:>7.4f} {m['dogruluk']:>9.4f} "
          f"{m['duyarlilik_ozg90']:>10.4f} {m['brier']:>7.4f}")
    satirlar.append(dict(kume=ad, **m))

pd.DataFrame(satirlar).to_csv("sonuclar/sonuc_pilot.csv", index=False)
ic = satirlar[0]["auc"]
print(f"\nFARK (ic - dis test): {ic - satirlar[2]['auc']:+.4f} AUC")
print("sonuc_pilot.csv yazildi")
