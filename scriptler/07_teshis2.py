"""
5. BLOK, IKINCI TUR: renk elendiyse imza nerede?

Blok 6 sonucu: merkez imzasi renkte (6 sayi %86) AMA rengi bozmak da silmek de
basarim dususunu kapatmadi. Yani imzanin yeri ile kaybin sebebi ayni sey degil.
Bu script "renk disinda ne var" sorusunun ilk iki testini yapar.

  A) GRI goruntuden merkez tahmini. Renk silindiginde merkez hala bilinebiliyorsa
     imza dokuda ve kontrast yapisinda da var demektir.
       A1: gri istatistiklerden (2 sayi: luma ort + std) lojistik regresyon
       A2: CNN, gri goruntu
  B) ODAK / KESKINLIK. Karo basina Laplacian varyansi. Tarayici ve buyutme farki
     en cok burada gorunur.
       B1: merkez basina keskinlik dagilimi
       B2: sadece keskinlik istatistiklerinden merkez tahmini

Karsilastirma tabani (04_teshis.py, ayni tohum ve ayni hasta ayrimi):
  sans 0,2000 | renk 6 sayi 0,8605 | CNN renkli 0,9445
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torchvision.models import resnet18, ResNet18_Weights

TOHUM = 42
MERKEZLER = [0, 1, 2, 3, 4]
ORNEK_BASI = 4_000
TUR = 4
YIGIN = 128

rng = np.random.default_rng(TOHUM)
torch.manual_seed(TOHUM); np.random.seed(TOHUM)
aygit = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X = np.load("veri/altkume/X.npy", mmap_mode="r")
md = pd.read_parquet("veri/altkume/secim.parquet").reset_index(drop=True)
md["idx"] = np.arange(len(md))

# ---- 04_teshis.py ile AYNI ornekleme ve AYNI hasta ayrimi -----------------
parcalar = []
for m in MERKEZLER:
    g = md[md.center == m]
    for e in (0, 1):
        alt = g[g.label == e]
        parcalar.append(alt.sample(n=min(ORNEK_BASI // 2, len(alt)), random_state=TOHUM))
ornek = pd.concat(parcalar).sort_values("idx").reset_index(drop=True)

egitim_h, test_h = [], []
for m in MERKEZLER:
    h = np.sort(ornek[ornek.center == m].patient.unique())
    karisik = rng.permutation(h)
    kes = max(1, int(len(h) * 0.7))
    egitim_h += list(karisik[:kes]); test_h += list(karisik[kes:])
egitim = ornek[ornek.patient.isin(egitim_h)]
test = ornek[ornek.patient.isin(test_h)]
print(f"ornek {len(ornek):,} karo | egitim {len(egitim):,} ({len(egitim_h)} hasta) | "
      f"test {len(test):,} ({len(test_h)} hasta)")

LAP = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                   device=aygit).view(1, 1, 3, 3)


def ozellikler(d):
    """Karo basina: luma ort, luma std, Laplacian varyansi, yuksek frekans enerjisi."""
    idx = np.sort(d.idx.values)
    cikti = np.zeros((len(idx), 4), dtype=np.float32)
    for i in range(0, len(idx), 1000):
        blok = np.asarray(X[idx[i:i + 1000]]).astype(np.float32) / 255.0
        t = torch.from_numpy(blok).to(aygit).permute(0, 3, 1, 2).contiguous()
        luma = (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]).unsqueeze(1)
        lap = F.conv2d(luma, LAP)
        cikti[i:i + 1000, 0] = luma.mean(dim=(1, 2, 3)).cpu().numpy()
        cikti[i:i + 1000, 1] = luma.std(dim=(1, 2, 3)).cpu().numpy()
        cikti[i:i + 1000, 2] = lap.var(dim=(1, 2, 3)).cpu().numpy()
        cikti[i:i + 1000, 3] = lap.abs().mean(dim=(1, 2, 3)).cpu().numpy()
    return cikti, d.set_index("idx").loc[idx].reset_index()


def lojistik(oz_e, y_e, oz_t, y_t, ad):
    ol = StandardScaler().fit(oz_e)
    lr = LogisticRegression(max_iter=2000).fit(ol.transform(oz_e), y_e)
    tah = lr.predict(ol.transform(oz_t))
    acc = accuracy_score(y_t, tah)
    print(f"{ad:>42}: {acc:.4f}")
    return acc, tah


oz_e, e_s = ozellikler(egitim)
oz_t, t_s = ozellikler(test)
ye, yt = e_s.center.values, t_s.center.values

print("\n" + "=" * 74 + "\nB1) MERKEZ BASINA KESKINLIK / ODAK\n" + "=" * 74)
tum, tum_s = ozellikler(ornek)
tum_s["luma"], tum_s["luma_std"] = tum[:, 0], tum[:, 1]
tum_s["lap_var"], tum_s["hf"] = tum[:, 2], tum[:, 3]
print(f"{'merkez':>7} {'luma ort':>9} {'luma std':>9} {'lap var':>10} {'yuksek frek':>12}")
for m, g in tum_s.groupby("center"):
    print(f"{m:>7} {g.luma.mean():>9.4f} {g.luma_std.mean():>9.4f} "
          f"{g.lap_var.mean():>10.5f} {g.hf.mean():>12.5f}")
lv = tum_s.groupby("center").lap_var.mean()
print(f"\nen keskin merkez m{lv.idxmax()} ({lv.max():.5f}) | "
      f"en yumusak m{lv.idxmin()} ({lv.min():.5f}) | oran {lv.max()/lv.min():.2f}x")

print("\n" + "=" * 74 + "\nA1 / B2) TEK TEK OZELLIK GRUPLARINDAN MERKEZ TAHMINI\n" + "=" * 74)
print(f"{'yontem':>42}: dogruluk   (sans 0.2000)")
acc_gri, _ = lojistik(oz_e[:, :2], ye, oz_t[:, :2], yt, "A1  gri 2 sayi (luma ort+std)")
acc_kes, _ = lojistik(oz_e[:, 2:], ye, oz_t[:, 2:], yt, "B2  keskinlik 2 sayi (lap var + hf)")
acc_dort, _ = lojistik(oz_e, ye, oz_t, yt, "A1+B2  dort sayi birlikte")

print("\n" + "=" * 74 + "\nA2) CNN, GRI GORUNTU\n" + "=" * 74)
ort_t = torch.tensor(float(oz_e[:, 0].mean()), device=aygit).view(1, 1, 1, 1)
std_t = torch.tensor(float(oz_e[:, 1].mean()), device=aygit).view(1, 1, 1, 1)


def yigin_getir(d, karistir):
    idxler = d.idx.values
    sira = np.random.permutation(len(idxler)) if karistir else np.arange(len(idxler))
    hedef = d.set_index("idx").center.to_dict()
    for i in range(0, len(sira), YIGIN):
        alt = np.sort(idxler[sira[i:i + YIGIN]])
        x = torch.from_numpy(np.asarray(X[alt]).copy()).to(aygit)
        x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        luma = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        luma = (luma - ort_t) / std_t
        y = torch.tensor([hedef[a] for a in alt], device=aygit)
        yield luma.expand(-1, 3, -1, -1), y


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

model.eval(); tah, ger = [], []
with torch.no_grad():
    for x, y in yigin_getir(test, False):
        tah += model(x).argmax(1).cpu().tolist(); ger += y.cpu().tolist()
acc_cnn_gri = accuracy_score(ger, tah)
print(f"\nCNN (gri) test dogrulugu: {acc_cnn_gri:.4f}")
print(pd.DataFrame(confusion_matrix(ger, tah),
                   index=[f"m{m}" for m in MERKEZLER],
                   columns=[f"m{m}" for m in MERKEZLER]).to_string())

print("\n" + "=" * 74 + "\nHUKUM\n" + "=" * 74)
satirlar = [("sans seviyesi", 0.2000), ("A1  gri 2 sayi", acc_gri),
            ("B2  keskinlik 2 sayi", acc_kes), ("A1+B2  dort sayi", acc_dort),
            ("renk 6 sayi (Blok 5)", 0.8605), ("A2  CNN gri goruntu", acc_cnn_gri),
            ("CNN renkli goruntu (Blok 5)", 0.9445)]
for ad, v in sorted(satirlar, key=lambda x: x[1]):
    print(f"{ad:>30}: {v:.4f}")
pd.DataFrame(satirlar, columns=["yontem", "dogruluk"]).to_csv("sonuclar/sonuc_teshis2.csv", index=False)
print("\nsonuclar/sonuc_teshis2.csv yazildi")
