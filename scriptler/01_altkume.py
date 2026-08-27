"""
1. ADIM: ALT KUME CIKARMA. Bir kez calisir, sonraki butun kosular bunu kullanir.

Neden alt kume:
  455 bin karonun tamamiyla bes kat egitim bu makinede gereksiz uzun surer.
  Bizim sorumuz "en iyi model hangisi" degil, "ic ve dis olcum arasindaki
  fark ne". Fark, sabit bir butceyle de olculur; onemli olan her katta
  AYNI butcenin kullanilmasi.

Secim kurallari (uctu de bilincli):
  1. Merkez basina esit sayi        -> katlar arasi fark veriden gelmesin
  2. Etiket bazinda dengeli (50/50) -> veri seti zaten dengeli, bozmuyoruz
  3. Hasta bazinda orantili         -> tek hastanin slaytlari kumeye hakim olmasin
                                       (merkez basina sadece 7-10 hasta var)

Dosya sirasina guvenilmiyor: 21 dosyanin 6'si tek merkezli, 8'i tek etiketli.
"""
import io
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

TOHUM = 42
HAVUZ = 30_000          # merkez basina cekilecek karo (15k tumor + 15k tumor degil)
CIKTI = "veri/altkume"
rng = np.random.default_rng(TOHUM)

os.makedirs(CIKTI, exist_ok=True)
md = pd.read_parquet("metadata.parquet")

# 00 adiminda dosya ici satir sirasi korunarak birlestirilmisti; geri kazaniyoruz.
md["satir"] = md.groupby("dosya").cumcount()


def hasta_bazinda_sec(grup, hedef):
    """Bir merkez-etiket grubundan, hastalara orantili dagitarak hedef kadar sec."""
    hastalar = grup["patient"].unique()
    pay = {h: len(grup[grup.patient == h]) for h in hastalar}
    toplam = sum(pay.values())
    secilen = []
    for h in hastalar:
        n = int(round(hedef * pay[h] / toplam))
        alt = grup[grup.patient == h]
        n = min(n, len(alt))
        if n > 0:
            secilen.append(alt.sample(n=n, random_state=TOHUM))
    out = pd.concat(secilen) if secilen else grup.head(0)
    # yuvarlamadan dogan sapmayi duzelt
    if len(out) > hedef:
        out = out.sample(n=hedef, random_state=TOHUM)
    return out


print("=" * 74)
print("SECIM")
print("=" * 74)
secimler = []
for merkez in sorted(md.center.unique()):
    m = md[md.center == merkez]
    parca = []
    for etiket in (0, 1):
        g = m[m.label == etiket]
        s = hasta_bazinda_sec(g, HAVUZ // 2)
        parca.append(s)
    sec = pd.concat(parca)
    secimler.append(sec)
    print(f"merkez {merkez} ({m.merkez_adi.iloc[0]:>14}): "
          f"{len(sec):,} karo secildi, {sec.patient.nunique()} hastadan, "
          f"tumor orani %{sec.label.mean()*100:.1f}")

secim = pd.concat(secimler, ignore_index=True)
print(f"\ntoplam secilen: {len(secim):,}")

print("\n" + "=" * 74)
print("GORUNTULERI COZME  (dosya dosya, tek seferde tek dosya bellekte)")
print("=" * 74)
secim["global_id"] = np.arange(len(secim))
X = np.zeros((len(secim), 96, 96, 3), dtype=np.uint8)

for dosya, grup in secim.groupby("dosya"):
    tablo = pq.read_table(f"veri/data/{dosya}", columns=["image"])
    sutun = tablo.column("image").combine_chunks()
    baytlar = sutun.field("bytes")
    for gid, satir in zip(grup.global_id.values, grup.satir.values):
        im = Image.open(io.BytesIO(baytlar[int(satir)].as_py())).convert("RGB")
        X[gid] = np.asarray(im, dtype=np.uint8)
    print(f"  {dosya}: {len(grup):,} karo cozuldu")
    del tablo, sutun, baytlar

np.save(f"{CIKTI}/X.npy", X)
secim.drop(columns=["global_id"]).to_parquet(f"{CIKTI}/secim.parquet")

print("\n" + "=" * 74)
print("DOGRULAMA")
print("=" * 74)
print(f"X sekli      : {X.shape}  ({X.nbytes/1e9:.2f} GB, uint8)")
print(f"piksel araligi: {X.min()} - {X.max()}")
# NOT: std(axis=1) tum diziyi float64e cevirir (33 GB). Parca parca hesaplaniyor.
bos = sum(int((X[i:i+5000].reshape(-1, 96*96*3).astype(np.float32).std(axis=1) < 1.0).sum())
          for i in range(0, len(X), 5000))
print(f"neredeyse duz (bos) karo sayisi: {bos}")
print(f"\nyazildi: {CIKTI}/X.npy  ve  {CIKTI}/secim.parquet")
