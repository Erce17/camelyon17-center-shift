[← README](../README.md)

# Müdahale: rengi bozmak ya da silmek düşüşü kapatıyor mu?

Teşhis "imza renkte" dedi. `05_mudahale.py` bunu iki kolla test eder: eğitimde
rastgele renk oynatma (`--kol renk_artirma`), ve rengi tamamen silme
(`--kol gri`). Taban çizgisiyle aynı tohumlar, aynı hastalar, aynı bütçe.

**Tuzak baştan işaretlendi:** düşüşü sıfırlamanın en kolay yolu modeli
kötüleştirmektir — her yerde eşit kötüyse fark kalmaz ve müdahale başarılı görünür.
O yüzden iki eksene birden bakılır: düşüş, **ve** dış test mutlak başarımı.

| kol | iç AUC | dış AUC | düşüş | ± | poz | dış duy | dış brier |
|---|---|---|---|---|---|---|---|
| taban çizgisi | 0,9726 | 0,9482 | +0,0244 | 0,0223 | 14/15 | 0,8579 | 0,1248 |
| renk artırma | 0,9763 | 0,9513 | +0,0250 | 0,0202 | 13/15 | 0,8662 | 0,1190 |
| gri tonlama | 0,9634 | 0,9394 | +0,0240 | 0,0498 | 11/15 | 0,8441 | 0,1305 |

**Hiçbir müdahale düşüşü kapatmadı.** Eşli farklar (+0,0006 ± 0,0138 ve
-0,0004 ± 0,0599) tamamen gürültü içinde.

**Renk artırma** düşüşe dokunmadı ama modeli her yerde hafif yukarı çekti (dış AUC
+0,0031, dış duyarlılık +0,0083, dış brier -0,0058). Zararsız bir düzenlileştirme,
kayma çözümü değil. Kazançlar gürültü sınırında; üstüne iddia kurulmaz.

**Gri tonlamanın asıl hikâyesi kat bazında:**

| kat | taban çizgisi | renk artırma | gri tonlama |
|---|---|---|---|
| 0 | 0,0389 | 0,0237 | **-0,0090** |
| 1 | 0,0341 | 0,0385 | 0,0198 |
| 2 | 0,0265 | 0,0349 | 0,0055 |
| 3 | 0,0209 | 0,0254 | **-0,0006** |
| 4 | 0,0016 | 0,0024 | **+0,1043** |

Renk silinince **dört katta düşüş kayboldu**, ama **kat 4'te taban çizgisinin 50
katına fırladı.** Ortalamanın sabit kalmasının tek sebebi bu tek kat, ve standart
sapmanın 0,0223'ten 0,0498'e çıkması da bundan.

Açıklaması teşhis tablosunda duruyor: merkez 4 sadece en açık değil, **en düşük
kontrastlı** merkez (std 0,109 / 0,130 / 0,093; diğerleri 0,145 / 0,165 / 0,130
bandında). Renk atıldığında elde yalnızca parlaklık farkları kalıyor ve m4'ün
sıkışık kontrast aralığı modeli çökertiyor.

### Bu deponun asıl sonucu

> **Merkezin renkten ayırt edilebilir olması, başarım düşüşünün renkten geldiği
> anlamına gelmez.**

İki ayrı şey. Model hangi hastaneden geldiğini renge bakarak %86 doğrulukla
biliyor. Ama tümör tespitinde dış merkezde kaybettiği şey renk değil: renk tamamen
silindi, kayıp durmadı.

"Boya normalizasyonu yapın, kayma çözülür" bu alanda sıkça tekrarlanan bir tavsiye.
Bu veride çalışmıyor, ve hangi merkezde ters teptiği ölçülmüş durumda.

**Açık soru:** renk değilse ne? Aday sebepler doku ve morfoloji farkı, tarayıcı
çözünürlüğü ve odak, hasta popülasyonu.

---
