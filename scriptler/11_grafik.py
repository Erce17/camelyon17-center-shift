"""
10. BLOK: GORSELLESTIRME.

Depoda sadece tablo vardi. Uc grafik uretir:

  gorseller/01_kalibrasyon.png  reliability diagram. Kalibrasyon hatasinin YONUNU
                                gosterir: model dis merkezde tumorlere gereginden
                                DUSUK olasilik veriyor (egri capraz cizginin ustunde),
                                ve BN uyarlamasi bunu capraz cizgiye yaklastiriyor.
  gorseller/02_kat_auc.png      kat bazinda dis AUC: taban vs etiketsiz BN uyarlamasi
  gorseller/03_acik.png         acigin yuzde kaci kapandi, mudahale x olcut

Renkler: dataviz referans paletinin sabit sirali ilk uc slotu.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # kategorik slot 1,2,3
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4de"
Path("gorseller").mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
})


def kalibrasyon_egrisi(skor, y, kova=10):
    """Reliability diagram noktalari: tahmin edilen olasilik vs gozlenen oran."""
    kenar = np.linspace(0, 1, kova + 1)
    xs, ys, n = [], [], []
    for i in range(kova):
        m = (skor >= kenar[i]) & (skor < kenar[i + 1] if i < kova - 1 else skor <= 1.0)
        if m.sum() < 5:
            continue
        xs.append(skor[m].mean()); ys.append(y[m].mean()); n.append(int(m.sum()))
    return np.array(xs), np.array(ys), n


# ---------------------------------------------------------------- 1. kalibrasyon
sk = pd.read_csv("sonuclar/skorlar_kat0_tohum42.csv")
uy = pd.read_csv("sonuclar/uyarlama_merkez0.csv")
uy = uy[uy.rol == "degerlendirme"] if "rol" in uy.columns else uy   # uyarlama satirlari skorsuz
ic = sk[sk.kume == "ic_dogrulama"]

fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.plot([0, 1], [0, 1], "--", color=INK2, lw=1, zorder=1, label="kusursuz kalibrasyon")
for (s, y, ad, renk) in [
        (ic.skor.values, ic.etiket.values, "ic dogrulama (egitim merkezleri)", S1),
        (uy.skor_once.values, uy.etiket.values, "dis merkez, uyarlamasiz", S2),
        (uy.skor_sonra.values, uy.etiket.values, "dis merkez, BN uyarlamasi sonrasi", S3)]:
    xs, ys, _ = kalibrasyon_egrisi(s, y)
    ax.plot(xs, ys, "-o", color=renk, lw=2, ms=6, mec="white", mew=1.5, label=ad, zorder=3)

ax.annotate("capraz cizgiye ne kadar yakinsa\nkalibrasyon o kadar iyi",
            xy=(0.62, 0.52), xytext=(0.66, 0.28), fontsize=8.5, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK2, lw=0.9))
ax.set_xlabel("modelin verdigi olasilik")
ax.set_ylabel("gercekte tumor cikma orani")
ax.set_title("Model yeni hastanede tumorlerden emin olamiyor\n"
             "egri capraz cizginin USTUNDE: model gereginden DUSUK olasilik veriyor",
             loc="left", color=INK)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.grid(True, lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig("gorseller/01_kalibrasyon.png", dpi=160)
print("gorseller/01_kalibrasyon.png")

# ---------------------------------------------------------------- 2. kat bazinda AUC
bn = pd.read_csv("sonuclar/sonuc_bn.csv")
p = bn.pivot_table(index="kat", columns="kol", values="auc")
katlar = np.arange(5); g = 0.36

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.bar(katlar - g/2 - 0.01, p["taban"], g, color=S2, label="uyarlamasiz", zorder=3)
ax.bar(katlar + g/2 + 0.01, p["bn_tam"], g, color=S3, label="BN uyarlamasi (etiketsiz)", zorder=3)
for i, k in enumerate(katlar):
    f = p["bn_tam"][k] - p["taban"][k]
    ax.text(k, max(p["taban"][k], p["bn_tam"][k]) + 0.004, f"{f:+.3f}",
            ha="center", fontsize=9, color=INK if f > 0 else S2)
ax.set_xticks(katlar, [f"m{k} disarida" for k in katlar])
ax.set_ylabel("dis test AUC"); ax.set_ylim(0.88, 1.0)
ax.set_title("Etiketsiz uyarlama her merkezde ise yaramiyor\n"
             "5 katin 3'unde kazanc, 2'sinde kayip", loc="left", color=INK)
ax.grid(True, axis="y", lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig("gorseller/02_kat_auc.png", dpi=160)
print("gorseller/02_kat_auc.png")

# ---------------------------------------------------------------- 3. acik kapanma
t = pd.read_csv("sonuclar/sonuc_tam.csv")
pt = t.pivot_table(index=["kat", "tohum"], columns="kume",
                   values=["auc", "dogruluk", "brier", "duyarlilik_ozg90"])
ic_ref = {m: pt[m]["ic_dogrulama"].mean() for m in ["auc", "dogruluk", "brier", "duyarlilik_ozg90"]}
tb = bn[bn.kol == "taban"]
olcut = [("auc", "AUC"), ("dogruluk", "dogruluk"),
         ("duyarlilik_ozg90", "duyarlilik"), ("brier", "kalibrasyon\n(brier)")]

kollar = [("esik200", "esik tasima\n(200 ETIKETLI)", S2), ("bn_tam", "BN uyarlamasi\n(ETIKETSIZ)", S3)]
x = np.arange(len(olcut)); g = 0.36
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for i, (kol, ad, renk) in enumerate(kollar):
    gg = bn[bn.kol == kol]
    v = [(gg[m].mean() - tb[m].mean()) / (ic_ref[m] - tb[m].mean()) * 100 for m, _ in olcut]
    ax.bar(x + (i - 0.5) * (g + 0.02), v, g, color=renk, label=ad, zorder=3)
    for xi, vi in zip(x + (i - 0.5) * (g + 0.02), v):
        ax.text(xi, vi + 2, f"%{vi:.0f}", ha="center", fontsize=9, color=INK)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(x, [ad for _, ad in olcut])
ax.set_ylabel("acigin yuzde kaci kapandi")
ax.set_ylim(-8, 112)
ax.set_title("Etiketsiz mudahale, etiketliden daha fazla is goruyor\n"
             "esik tasima AUC'ye ve kalibrasyona dokunamiyor", loc="left", color=INK)
ax.grid(True, axis="y", lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig("gorseller/03_acik.png", dpi=160)
print("gorseller/03_acik.png")
