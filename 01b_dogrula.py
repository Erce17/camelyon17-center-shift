"""Alt kume dogrulamasi. X.npy diske yazildi, bellege almadan kontrol ediyoruz."""
import numpy as np
import pandas as pd

X = np.load("veri/altkume/X.npy", mmap_mode="r")   # bellege kopyalanmiyor
secim = pd.read_parquet("veri/altkume/secim.parquet")

print(f"X sekli       : {X.shape}   ({X.nbytes/1e9:.2f} GB, {X.dtype})")
print(f"secim satiri  : {len(secim):,}")
print(f"eslesme       : {'TAMAM' if len(X) == len(secim) else 'BOZUK'}")

print("\nmerkez bazinda secilen:")
ozet = secim.groupby(["center", "merkez_adi"]).agg(
    karo=("label", "size"), hasta=("patient", "nunique"), tumor=("label", "mean"))
ozet["tumor"] = (ozet["tumor"] * 100).round(1)
print(ozet.to_string())

print("\nhasta basina karo (en az / en cok):")
h = secim.groupby("patient").size()
print(f"  en az {h.min():,}  en cok {h.max():,}  ortanca {int(h.median()):,}")

# Parca parca: bellekte ayni anda en fazla 5000 karo
duz, toplam_std, n = 0, 0.0, 0
for i in range(0, len(X), 5000):
    parca = np.asarray(X[i:i+5000]).reshape(-1, 96*96*3).astype(np.float32)
    s = parca.std(axis=1)
    duz += int((s < 1.0).sum())
    toplam_std += float(s.sum()); n += len(s)
print(f"\npiksel araligi        : {int(np.asarray(X[:1000]).min())} - {int(np.asarray(X[:1000]).max())}")
print(f"neredeyse duz karo    : {duz}  ({duz/len(X)*100:.3f}%)")
print(f"ortalama karo std     : {toplam_std/n:.1f}")
print("\nDuz karo yoksa: WILDS zaten doku iceren karolari secmis, bos cam elenmis.")
