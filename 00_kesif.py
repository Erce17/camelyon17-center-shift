"""
0. ADIM: VERIYI TANIMAK. Model yok, egitim yok, tek satir tahmin yok.

Parquet'ten SADECE metadata sutunlarini okur, goruntu sutununa dokunmaz.
Parquet sutun bazli sakladigi icin 455 bin satirlik tablo saniyeler icinde gelir.

Cevaplanacak sorular:
  1. Kac merkez, her merkezde kac hasta / slayt / karo?
  2. Tumor orani merkezden merkeze degisiyor mu?  -> kovaryat mi, etiket kaymasi mi
  3. Hangi merkez hangi bolumde?                  -> ezberden degil dosyadan
  4. Hasta veya slayt bolumler arasinda tasiyor mu? -> sizinti kontrolu
  5. Karolar dosya icinde sirali mi?               -> alt kume nasil secilecek
"""
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

KOK = "veri/data"
SUTUNLAR = ["label", "center", "patient", "node", "x_coord", "y_coord", "slide"]

# center sutunu HF ClassLabel: sayi -> rol adi. Parquet metadatasindan geliyor.
MERKEZ_ADI = {
    0: "egitim-1", 1: "DIS-dogrulama", 2: "DIS-test", 3: "egitim-2", 4: "egitim-3",
}

parcalar = []
for yol in sorted(glob.glob(f"{KOK}/*.parquet")):
    bolum = os.path.basename(yol).split("-")[0]      # train / validation / test
    df = pq.read_table(yol, columns=SUTUNLAR).to_pandas()
    df["bolum"] = bolum
    df["dosya"] = os.path.basename(yol)
    parcalar.append(df)

md = pd.concat(parcalar, ignore_index=True)
md["merkez_adi"] = md["center"].map(MERKEZ_ADI)

print("=" * 74)
print("1. HAM YAPI")
print("=" * 74)
print(f"toplam karo : {len(md):,}")
print(f"dosya       : {len(parcalar)}")
print(f"hasta       : {md.patient.nunique()}   slayt: {md.slide.nunique()}   merkez: {md.center.nunique()}")
print(f"tumor orani (genel): {md.label.mean()*100:.2f}%")

print("\n" + "=" * 74)
print("2. MERKEZ BAZINDA  (projenin butun konusu bu tablo)")
print("=" * 74)
ozet = md.groupby(["center", "merkez_adi"]).agg(
    hasta=("patient", "nunique"),
    slayt=("slide", "nunique"),
    karo=("label", "size"),
    tumor_orani=("label", "mean"),
)
ozet["tumor_orani"] = (ozet["tumor_orani"] * 100).round(2)
ozet["karo"] = ozet["karo"].map("{:,}".format)
print(ozet.to_string())

oranlar = md.groupby("center")["label"].mean() * 100
print(f"\ntumor orani araligi: %{oranlar.min():.2f} - %{oranlar.max():.2f}"
      f"  (fark {oranlar.max()-oranlar.min():.2f} puan)")
print("Okuma notu:")
print("  Fark kucukse  -> agirlikli olarak KOVARYAT kaymasi (renk/tarayici).")
print("                   Cozum: renk normalizasyonu, renk artirma.")
print("  Fark buyukse  -> isin icinde ETIKET kaymasi da var (sevk deseni).")
print("                   Cozum: esik ve prior duzeltmesi. Renk ise yaramaz.")

print("\n" + "=" * 74)
print("3. BOLUNME: HANGI MERKEZ NEREDE")
print("=" * 74)
print(pd.crosstab(md["bolum"], md["merkez_adi"]).to_string())

print("\n" + "=" * 74)
print("4. SIZINTI KONTROLLERI")
print("=" * 74)
print(f"birden fazla merkezde gorunen hasta : {(md.groupby('patient')['center'].nunique() > 1).sum()}")
print(f"birden fazla merkezde gorunen slayt : {(md.groupby('slide')['center'].nunique() > 1).sum()}")
print(f"hasta basina ortalama karo          : {len(md)/md.patient.nunique():,.0f}")
print(f"slayt basina ortalama karo          : {len(md)/md.slide.nunique():,.0f}")
print("Not: ayni slayttan gelen karolar bagimsiz degildir (komsu koordinatlar).")
print("     Bolunme hasta bazinda yapilmazsa ic olcum sisirilmis cikar.")

print("\n" + "=" * 74)
print("5. DOSYA ICI SIRALAMA  (alt kume secimini belirler)")
print("=" * 74)
ilk = parcalar[0]
print(f"ornek dosya: {ilk.dosya.iloc[0]}")
print(f"  icindeki merkez sayisi : {ilk.center.nunique()}   etiket sayisi: {ilk.label.nunique()}")
kirilim = md.groupby("dosya").agg(merkez=("center", "nunique"), etiket=("label", "nunique"))
print(f"\ntek merkezli dosya sayisi : {(kirilim.merkez == 1).sum()} / {len(kirilim)}")
print(f"tek etiketli dosya sayisi : {(kirilim.etiket == 1).sum()} / {len(kirilim)}")
print("Sonuc: dosyalar sirali. Alt kume dosya sirasindan DEGIL, global")
print("       karistirma + merkez ve etiket bazinda katmanli secimle alinacak.")

md.to_parquet("metadata.parquet")
print(f"\nmetadata.parquet yazildi ({len(md):,} satir, sonraki adimlar bunu kullanacak)")
