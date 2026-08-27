"""
5. BLOK: TESHIS. Dusus var (03_tam.py: +0,0244 AUC). Peki nereden geliyor?

Uc asama, en ucuzdan pahaliya. Sira onemli: pahali olan once kosulursa ucuz olanin
zaten yettigi gorulmez.

  A) Merkez basina renk istatistikleri. Hastaneler gercekten farkli renkte mi?
  B) SADECE 6 sayidan (RGB ortalama + std) merkez tahmini, lojistik regresyon.
     Bu yetiyorsa merkez imzasi boyadadir, dokuda degil.
  C) CNN merkez siniflandiricisi. Karonun kendisinden merkez tahmini.
     B'den belirgin iyiyse imza dokuda da var demektir.

Sans seviyesi %20 (bes merkez, dengeli ornekleme).
Her asamada ayrim HASTA bazinda: ayni hastanin karolari hem egitimde hem testte
olursa model merkezi degil hastayi ezberler.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18, ResNet18_Weights

TOHUM = 42
MERKEZLER = [0, 1, 2, 3, 4]
ORNEK_BASI = 4_000      # merkez basina karo
TUR = 4
YIGIN = 128

rng = np.random.default_rng(TOHUM)
torch.manual_seed(TOHUM); np.random.seed(TOHUM)
aygit = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X = np.load("veri/altkume/X.npy", mmap_mode="r")
md = pd.read_parquet("veri/altkume/secim.parquet").reset_index(drop=True)
md["idx"] = np.arange(len(md))

# ---- ornekleme: merkez basina esit, etiket dengeli -------------------------
parcalar = []
for m in MERKEZLER:
    g = md[md.center == m]
    for e in (0, 1):
        alt = g[g.label == e]
        parcalar.append(alt.sample(n=min(ORNEK_BASI // 2, len(alt)), random_state=TOHUM))
ornek = pd.concat(parcalar).sort_values("idx").reset_index(drop=True)
print(f"ornek: {len(ornek):,} karo | merkez basina {ornek.groupby('center').size().tolist()}")

# ---- HASTA bazli ayrim -----------------------------------------------------
egitim_h, test_h = [], []
for m in MERKEZLER:
    h = np.sort(ornek[ornek.center == m].patient.unique())
    karisik = rng.permutation(h)
    kes = max(1, int(len(h) * 0.7))
    egitim_h += list(karisik[:kes]); test_h += list(karisik[kes:])
egitim = ornek[ornek.patient.isin(egitim_h)]
test = ornek[ornek.patient.isin(test_h)]
print(f"egitim {len(egitim):,} karo / {len(egitim_h)} hasta | "
      f"test {len(test):,} karo / {len(test_h)} hasta")

# ---- ozellik cikarma: karo basina 6 sayi -----------------------------------
def renk_ozellik(d):
    idx = np.sort(d.idx.values)
    ozellik = np.zeros((len(idx), 6), dtype=np.float32)
    for i in range(0, len(idx), 2000):
        blok = np.asarray(X[idx[i:i + 2000]]).astype(np.float32) / 255.0
        ozellik[i:i + 2000, :3] = blok.mean(axis=(1, 2))
        ozellik[i:i + 2000, 3:] = blok.std(axis=(1, 2))
    return ozellik, d.set_index("idx").loc[idx].reset_index()

print("\n" + "=" * 74 + "\nA) MERKEZ BASINA RENK ISTATISTIKLERI\n" + "=" * 74)
oz_hepsi, ornek_sirali = renk_ozellik(ornek)
ornek_sirali["r"], ornek_sirali["g"], ornek_sirali["b"] = oz_hepsi[:, 0], oz_hepsi[:, 1], oz_hepsi[:, 2]
ornek_sirali["sr"], ornek_sirali["sg"], ornek_sirali["sb"] = oz_hepsi[:, 3], oz_hepsi[:, 4], oz_hepsi[:, 5]
print(f"{'merkez':>7} {'R ort':>7} {'G ort':>7} {'B ort':>7} {'R std':>7} {'G std':>7} {'B std':>7}")
for m, g in ornek_sirali.groupby("center"):
    print(f"{m:>7} {g.r.mean():>7.4f} {g.g.mean():>7.4f} {g.b.mean():>7.4f} "
          f"{g.sr.mean():>7.4f} {g.sg.mean():>7.4f} {g.sb.mean():>7.4f}")
kanal = ornek_sirali.groupby("center")[["r", "g", "b"]].mean()
print(f"\nmerkezler arasi en buyuk ortalama farki: "
      f"{(kanal.max() - kanal.min()).max():.4f} (kanal: {(kanal.max()-kanal.min()).idxmax()})")

print("\n" + "=" * 74 + "\nB) SADECE 6 SAYIDAN MERKEZ TAHMINI (lojistik regresyon)\n" + "=" * 74)
oz_e, e_s = renk_ozellik(egitim)
oz_t, t_s = renk_ozellik(test)
ol = StandardScaler().fit(oz_e)
lr = LogisticRegression(max_iter=2000)
lr.fit(ol.transform(oz_e), e_s.center.values)
tah_lr = lr.predict(ol.transform(oz_t))
acc_lr = accuracy_score(t_s.center.values, tah_lr)
print(f"test dogrulugu: {acc_lr:.4f}   (sans seviyesi 0.2000)")
print("karisiklik matrisi (satir gercek, sutun tahmin):")
print(pd.DataFrame(confusion_matrix(t_s.center.values, tah_lr),
                   index=[f"m{m}" for m in MERKEZLER],
                   columns=[f"m{m}" for m in MERKEZLER]).to_string())

print("\n" + "=" * 74 + "\nC) CNN MERKEZ SINIFLANDIRICISI\n" + "=" * 74)
ort_t = torch.tensor(oz_e[:, :3].mean(axis=0), device=aygit).view(1, 3, 1, 1)
std_t = torch.tensor(oz_e[:, 3:].mean(axis=0), device=aygit).view(1, 3, 1, 1)

def yigin_getir(d, karistir):
    idxler = d.idx.values
    sira = np.random.permutation(len(idxler)) if karistir else np.arange(len(idxler))
    hedef = d.set_index("idx").center.to_dict()
    for i in range(0, len(sira), YIGIN):
        alt = np.sort(idxler[sira[i:i + YIGIN]])
        x = torch.from_numpy(np.asarray(X[alt]).copy()).to(aygit)
        x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        y = torch.tensor([hedef[a] for a in alt], device=aygit)
        yield (x - ort_t) / std_t, y

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 5)
model = model.to(aygit)
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
kayip_fn = nn.CrossEntropyLoss()

for tur in range(1, TUR + 1):
    model.train(); toplam, n = 0.0, 0
    for x, y in yigin_getir(egitim, True):
        opt.zero_grad(); k = kayip_fn(model(x), y); k.backward(); opt.step()
        toplam += k.item() * len(y); n += len(y)
    model.eval(); dogru, top = 0, 0
    with torch.no_grad():
        for x, y in yigin_getir(test, False):
            dogru += (model(x).argmax(1) == y).sum().item(); top += len(y)
    print(f"tur {tur}: kayip {toplam/n:.4f} | test dogrulugu {dogru/top:.4f}")

model.eval(); tahminler, gercekler = [], []
with torch.no_grad():
    for x, y in yigin_getir(test, False):
        tahminler += model(x).argmax(1).cpu().tolist(); gercekler += y.cpu().tolist()
acc_cnn = accuracy_score(gercekler, tahminler)
print(f"\nCNN test dogrulugu: {acc_cnn:.4f}")
print("karisiklik matrisi (satir gercek, sutun tahmin):")
print(pd.DataFrame(confusion_matrix(gercekler, tahminler),
                   index=[f"m{m}" for m in MERKEZLER],
                   columns=[f"m{m}" for m in MERKEZLER]).to_string())

print("\n" + "=" * 74 + "\nHUKUM\n" + "=" * 74)
print(f"sans 0.2000 | 6 sayi (renk) {acc_lr:.4f} | CNN (goruntu) {acc_cnn:.4f}")
print(f"CNN'in renk uzerine kattigi: {acc_cnn - acc_lr:+.4f}")
pd.DataFrame([dict(yontem="sans", dogruluk=0.2),
              dict(yontem="renk_6_sayi_lojistik", dogruluk=acc_lr),
              dict(yontem="cnn_goruntu", dogruluk=acc_cnn)]).to_csv("sonuclar/sonuc_teshis.csv", index=False)
print("sonuc_teshis.csv yazildi")
