"""
YENI BIR MERKEZE UYARLAMA.

Bu deponun urun girisi. Senaryo: elinizde egitilmis bir model var, yeni bir
hastaneye kurulum yapiyorsunuz. Bu script o hastanenin verisiyle modeli uyarlar
ve oncesi/sonrasi raporu basar.

    python uyarla.py --model modeller/kat0_tohum42.pt --merkez 0
    python uyarla.py --model modeller/kat0_tohum42.pt --merkez 0 --etiketli 200

Iki mudahale uygulanir, ikisi de modeli YENIDEN EGITMEZ:

  1. BatchNorm uyarlamasi (ETIKETSIZ). Hedef merkezin ham goruntuleri modelden
     bir kez gecirilir, 20 BN katmaninin ortalama/varyansi o merkeze tasinir.
     Olculen etki: AUC aciginin %67'si, kalibrasyon aciginin %98,7'si kapanir.
     >> En az 400-800 karo verin. 100 karo ile ZARAR VERIR (brier %311 kotulesir).

  2. Esik tasima (ETIKETLI, istege bagli). --etiketli N verilirse hedef merkezden
     N etiketli karo ile karar esigi yeniden bulunur.
     Olculen etki: dogruluk aciginin %37 (N=50) ile %52'si (N=200).

SIZINTI KURALI: uyarlama ve kalibrasyon karolari, degerlendirme yapilan hastalardan
secilmez. Script bunu kendisi ayirir ve raporda hangi hastalarin nerede kullanildigini
yazar.

Ayrintili olcumler: docs/05-kalibrasyon.md ve docs/06-etiketsiz-uyarlama.md
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from torchvision.models import resnet18

HASTA_BASI_SINIF = 150
OLCUM_HASTA = 3
YIGIN = 128

ap = argparse.ArgumentParser(description="Yeni bir merkeze uyarlama")
ap.add_argument("--model", required=True, help="modeller/*.pt")
ap.add_argument("--merkez", type=int, required=True, help="hedef merkez kimligi")
ap.add_argument("--etiketli", type=int, default=0,
                help="hedef merkezden kac ETIKETLI karo kullanilsin (0 = sadece BN)")
ap.add_argument("--bn-karo", type=int, default=800, help="etiketsiz uyarlama karo sayisi")
ap.add_argument("--tohum", type=int, default=42)
ap.add_argument("--cikti", default=None, help="skorlarin yazilacagi csv")
args = ap.parse_args()

aygit = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
rng = np.random.default_rng(args.tohum)
torch.manual_seed(args.tohum); np.random.seed(args.tohum)

# ---- model ----
paket = torch.load(args.model, map_location=aygit, weights_only=False)
model = resnet18(weights=None)
model.fc = nn.Linear(512, paket.get("cikis", 1))
model.load_state_dict(paket["state_dict"])
model = model.to(aygit).eval()
temiz = {k: v.detach().clone() for k, v in model.state_dict().items()}
ort = np.array(paket["normalizasyon"]["ort"], dtype=np.float32)
std = np.array(paket["normalizasyon"]["std"], dtype=np.float32)
ort_t = torch.tensor(ort, device=aygit).view(1, 3, 1, 1)
std_t = torch.tensor(std, device=aygit).view(1, 3, 1, 1)

manifest_yolu = Path(args.model).with_suffix(".json")
manifest = json.loads(manifest_yolu.read_text()) if manifest_yolu.exists() else {}
egitim_merkezleri = manifest.get("bolunme", {}).get("egitim_merkezleri", "?")

# ---- hedef merkezin verisi ----
X = np.load("veri/altkume/X.npy", mmap_mode="r")
md = pd.read_parquet("veri/altkume/secim.parquet").reset_index(drop=True)
md["idx"] = np.arange(len(md))
hedef = md[md.center == args.merkez]
if len(hedef) == 0:
    raise SystemExit(f"merkez {args.merkez} icin karo bulunamadi")

_h = hedef.groupby("patient").label.agg(["size", "sum"])
_h["min_sinif"] = np.minimum(_h["sum"], _h["size"] - _h["sum"])
uygun = np.sort(_h[_h.min_sinif >= HASTA_BASI_SINIF].index.values)

# --- SIZINTI AYRIMI: degerlendirme hastalari vs uyarlama hastalari ---
deg_hastalar = rng.choice(uygun, min(OLCUM_HASTA, len(uygun)), replace=False)
uy_hastalar = [h for h in np.sort(hedef.patient.unique()) if h not in set(deg_hastalar)]

parcalar = []
for h in deg_hastalar:
    g = hedef[hedef.patient == h]
    t, n = g[g.label == 1], g[g.label == 0]
    k = min(HASTA_BASI_SINIF, len(t), len(n))
    parcalar += [t.sample(n=k, random_state=args.tohum), n.sample(n=k, random_state=args.tohum)]
degerlendirme = pd.concat(parcalar)

uy = hedef[hedef.patient.isin(uy_hastalar)]
_t, _n = uy[uy.label == 1], uy[uy.label == 0]
_k = min(len(_t), len(_n), args.bn_karo // 2)
uyarlama = (pd.concat([_t.sample(n=_k, random_state=args.tohum),
                       _n.sample(n=_k, random_state=args.tohum)])
            if _k > 0 else uy.iloc[0:0])


def yigin(idxler):
    for i in range(0, len(idxler), YIGIN):
        alt = np.sort(idxler[i:i + YIGIN])
        x = torch.from_numpy(np.asarray(X[alt]).copy()).to(aygit)
        x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        yield (x - ort_t) / std_t, alt


def skorla(d):
    model.eval(); skor = np.zeros(len(d), dtype=np.float32)
    yer = {v: i for i, v in enumerate(d.idx.values)}
    with torch.no_grad():
        for x, alt in yigin(d.idx.values):
            p = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            for j, a in enumerate(alt):
                skor[yer[a]] = p[j]
    return skor, d.label.values


def bn_uyarla(idxler):
    """ETIKETSIZ. Agirliklar degismez: gradyan yok, optimizer yok."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for x, _ in yigin(idxler):
            model(x)
    model.eval()


def esik_bul(skor, y, ozgulluk=0.90):
    neg = np.sort(skor[y == 0])
    return 0.5 if len(neg) == 0 else float(neg[min(int(len(neg) * ozgulluk), len(neg) - 1)])


def olc(skor, y, esik):
    fpr, tpr, _ = roc_curve(y, skor)
    return dict(auc=roc_auc_score(y, skor),
                dogruluk=float(((skor >= esik).astype(int) == y).mean()),
                duyarlilik=float(np.interp(0.10, fpr, tpr)),
                brier=brier_score_loss(y, skor))


print("=" * 78)
print(f"YENI MERKEZE UYARLAMA  ->  merkez {args.merkez}")
print("=" * 78)
print(f"model                 : {args.model}")
print(f"egitim merkezleri     : {egitim_merkezleri}")
print(f"degerlendirme         : {len(degerlendirme):,} karo | hastalar "
      f"{[int(h) for h in deg_hastalar]}")
print(f"uyarlama (ETIKETSIZ)  : {len(uyarlama):,} karo | hastalar "
      f"{[int(h) for h in uyarlama.patient.unique()]}")
print(f"sizinti kontrolu      : kesisim "
      f"{len(set(degerlendirme.patient) & set(uyarlama.patient))} hasta (0 olmali)")
if len(uyarlama) < 400:
    print(f"UYARI: uyarlama kumesi {len(uyarlama)} karo. 400'un altinda BN uyarlamasi "
          f"ZARAR verebilir (bkz. docs/06-etiketsiz-uyarlama.md)")

# --- 1. uyarlamasiz ---
s0, y0 = skorla(degerlendirme)
e0 = 0.5
if manifest.get("olcumler"):
    pass
m0 = olc(s0, y0, e0)

# --- 2. BN uyarlamasi ---
if len(uyarlama) > 0:
    bn_uyarla(uyarlama.idx.values)
    s1, y1 = skorla(degerlendirme)
    m1 = olc(s1, y1, e0)
else:
    s1, m1 = s0, m0

# --- 3. esik tasima (istege bagli, ETIKETLI) ---
m2 = None
if args.etiketli > 0 and len(uyarlama) > 0:
    s_uy, y_uy = skorla(uyarlama)
    n_al = min(args.etiketli // 2, len(s_uy) // 2)
    poz = np.where(y_uy == 1)[0][:n_al]; neg = np.where(y_uy == 0)[0][:n_al]
    sec = np.concatenate([poz, neg])
    e2 = esik_bul(s_uy[sec], y_uy[sec])
    m2 = olc(s1, y0, e2)
    pl = LogisticRegression(max_iter=1000).fit(s_uy[sec].reshape(-1, 1), y_uy[sec])
    s2 = pl.predict_proba(s1.reshape(-1, 1))[:, 1]
    m2["brier"] = brier_score_loss(y0, s2)

print("\n" + "=" * 78)
print(f"{'asama':>28} {'AUC':>8} {'dogruluk':>9} {'duyarlilik':>11} {'brier':>8}")
print("=" * 78)
print(f"{'1. uyarlamasiz':>28} {m0['auc']:>8.4f} {m0['dogruluk']:>9.4f} "
      f"{m0['duyarlilik']:>11.4f} {m0['brier']:>8.4f}")
print(f"{'2. + BN uyarlamasi (etiketsiz)':>28} {m1['auc']:>8.4f} {m1['dogruluk']:>9.4f} "
      f"{m1['duyarlilik']:>11.4f} {m1['brier']:>8.4f}")
if m2:
    print(f"{f'3. + esik tasima ({args.etiketli} etiketli)':>28} {m2['auc']:>8.4f} "
          f"{m2['dogruluk']:>9.4f} {m2['duyarlilik']:>11.4f} {m2['brier']:>8.4f}")
son = m2 or m1
print("=" * 78)
print(f"{'TOPLAM DEGISIM':>28} {son['auc']-m0['auc']:>+8.4f} "
      f"{son['dogruluk']-m0['dogruluk']:>+9.4f} {son['duyarlilik']-m0['duyarlilik']:>+11.4f} "
      f"{son['brier']-m0['brier']:>+8.4f}")
print("\n(brier'de eksi iyidir: kalibrasyon hatasi azalir)")

cikti = args.cikti or f"sonuclar/uyarlama_merkez{args.merkez}.csv"
pd.DataFrame(dict(idx=degerlendirme.idx.values, hasta=degerlendirme.patient.values,
                  etiket=y0, skor_once=s0, skor_sonra=s1)).to_csv(cikti, index=False)
print(f"\nkaro bazli skorlar -> {cikti}")
model.load_state_dict(temiz)
