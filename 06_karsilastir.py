"""
Uc kolu yan yana koyar: taban cizgisi, renk artirma, gri tonlama.

Iki eksene birden bakilir. Dusus tek basina yanilticidir: modeli kotulestirirsen
her yerde esit kotu olur, dusus sifira iner, mudahale basarili gorunur.
Asil olcu DIS TEST mutlak basarimidir.
"""
import numpy as np
import pandas as pd

KOLLAR = [("taban cizgisi", "sonuc_tam.csv"),
          ("renk artirma", "sonuc_renk_artirma.csv"),
          ("gri tonlama", "sonuc_gri.csv")]

tablo, dususler = [], {}
for ad, dosya in KOLLAR:
    try:
        d = pd.read_csv(dosya)
    except FileNotFoundError:
        print(f"[atlandi] {dosya} yok"); continue
    p = d.pivot_table(index=["kat", "tohum"], columns="kume",
                      values=["auc", "dogruluk", "brier", "duyarlilik_ozg90"])
    dusus = p["auc"]["ic_dogrulama"] - p["auc"]["dis_test"]
    dususler[ad] = dusus
    tablo.append(dict(
        kol=ad, n=len(d) // 3,
        ic_auc=p["auc"]["ic_dogrulama"].mean(),
        dis_auc=p["auc"]["dis_test"].mean(),
        dis_auc_std=p["auc"]["dis_test"].std(),
        dusus=dusus.mean(), dusus_std=dusus.std(),
        pozitif=f"{int((dusus > 0).sum())}/{len(dusus)}",
        dis_duy=p["duyarlilik_ozg90"]["dis_test"].mean(),
        dis_brier=p["brier"]["dis_test"].mean()))

t = pd.DataFrame(tablo)
print("=" * 92)
print("UC KOL YAN YANA")
print("=" * 92)
print(f"{'kol':>15} {'n':>3} {'ic AUC':>8} {'dis AUC':>9} {'DUSUS':>9} {'+-':>7} "
      f"{'poz':>6} {'dis duy':>8} {'dis brier':>10}")
for _, r in t.iterrows():
    print(f"{r.kol:>15} {r.n:>3} {r.ic_auc:>8.4f} {r.dis_auc:>9.4f} {r.dusus:>+9.4f} "
          f"{r.dusus_std:>7.4f} {r.pozitif:>6} {r.dis_duy:>8.4f} {r.dis_brier:>10.4f}")

if len(t) > 1:
    tb = t[t.kol == "taban cizgisi"].iloc[0]
    print("\n" + "=" * 92)
    print("TABAN CIZGISINE GORE DEGISIM")
    print("=" * 92)
    for _, r in t[t.kol != "taban cizgisi"].iterrows():
        dd = r.dusus - tb.dusus
        da = r.dis_auc - tb.dis_auc
        db = r.dis_brier - tb.dis_brier
        print(f"\n{r.kol.upper()}")
        print(f"  dusus     : {tb.dusus:+.4f} -> {r.dusus:+.4f}  ({dd:+.4f})"
              f"  {'KUCULDU' if dd < 0 else 'BUYUDU'}")
        print(f"  dis AUC   : {tb.dis_auc:.4f} -> {r.dis_auc:.4f}  ({da:+.4f})"
              f"  {'korundu/arttii' if da >= -0.005 else 'BEDEL VAR'}")
        print(f"  dis brier : {tb.dis_brier:.4f} -> {r.dis_brier:.4f}  ({db:+.4f})"
              f"  {'iyilesti' if db < 0 else 'kotulesti'}")
        # esli t-testi yerine, tohum-kat esli farklarin dagilimi
        ort = dususler[r.kol] - dususler["taban cizgisi"]
        print(f"  esli fark : {ort.mean():+.4f} +- {ort.std():.4f} "
              f"| 15 kosunun {int((ort < 0).sum())}'unde dusus azaldi")
        if da < -0.005 and dd < 0:
            print("  >> UYARI: dusus kuculdu ama dis basarim da dustu. "
                  "Model her yerde esit kotulesmis olabilir.")

print("\n" + "=" * 92)
print("KAT BASINA DUSUS")
print("=" * 92)
kat = pd.DataFrame({ad: d.groupby(level="kat").mean() for ad, d in dususler.items()})
print(kat.round(4).to_string())
kat.to_csv("karsilastirma_kat.csv")
t.to_csv("karsilastirma.csv", index=False)
print("\nkarsilastirma.csv ve karsilastirma_kat.csv yazildi")
