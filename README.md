# Camelyon17 — Merkez Kayması Ölçümü

Üç hastanenin verisiyle eğitilen bir patoloji modeli, hiç görmediği dördüncü
hastanede ne kaybeder, ve kayıp neyden gelir?

**Bu deponun çıktısı bir model değil, bir ölçüm.** Amaç en iyi sınıflandırıcıyı
kurmak değil; iç ve dış başarım arasındaki farkı savunulabilir biçimde ölçmek,
sonra o farkın kaynağını teşhis etmek.

---

## Sonuç, tek tabloda

15 koşu (5 kat x 3 tohum), her ölçüm 3 hasta ve 900 karo üzerinden.

| küme | AUC | doğruluk | duy@özg90 | brier |
|---|---|---|---|---|
| iç doğrulama | 0,9726 ± 0,0205 | 0,8896 | 0,9218 | 0,0873 |
| dış doğrulama | 0,9444 ± 0,0456 | 0,8444 | — | 0,1301 |
| dış test | 0,9482 ± 0,0206 | 0,8452 | 0,8579 | 0,1248 |

**Genel düşüş: +0,0244 ± 0,0223 AUC. 15 koşunun 14'ünde pozitif.**

### Asıl bulgu düşüşün varlığı değil, hangi metrikte olduğu

| metrik | göreli kayıp |
|---|---|
| AUC | %2,5 |
| doğruluk | %5,0 |
| duyarlılık @ özgüllük 0,90 | %6,9 |
| **brier (kalibrasyon)** | **%43,0** |

AUC en az kaybeden metrik. Model yeni merkeze taşındığında **sıralama yeteneğini
büyük ölçüde koruyor, ama eşiğini ve olasılık kalibrasyonunu kaybediyor.**

Ürün karşılığı: yeni bir hastaneye kurulum yaptığında AUC raporuna bakıp "sorun
yok" dersin. Aynı eşikle çalıştırdığında daha çok vaka kaçırırsın, ve modelin
verdiği olasılıklara artık güvenemezsin. **Eşik her merkezde yeniden ayarlanmalı.**

### Kayma tek boyutlu bir sabit değil

Kat başına düşüş: m0 +0,0389 · m1 +0,0341 · m2 +0,0265 · m3 +0,0209 · m4 +0,0016

Aynı koşuda iki farklı dış merkez arasındaki ortalama AUC farkı **0,0255** — genel
iç/dış farkı (0,0244) kadar büyük. Yani "hangi hastane dışarıda" etkisi, "içeride
mi dışarıda mı" etkisi kadar güçlü. **"Yeni hastanede şu kadar kaybedersiniz" diye
tek bir sayı vermek yanlış olur.**

---

## Teşhis: kayma nereden geliyor?

Merkez imzasını üç aşamada, en ucuz yöntemden başlayarak aradık (`04_teshis.py`).
Sıra önemli: pahalı olan önce koşulursa ucuz olanın zaten yettiği görülmez.

| yöntem | merkez tahmin doğruluğu |
|---|---|
| şans seviyesi | 0,2000 |
| karo başına **6 sayı** (RGB ortalama + std), lojistik regresyon | **0,8605** |
| CNN, tüm görüntü (96x96x3 = 27.648 piksel) | 0,9445 |

**Karo başına 6 sayı, 27.648 pikselin yaptığı işin %91'ini yapıyor.** Merkez imzası
ezici biçimde renkte: bir karonun hangi hastaneden geldiğini anlamak için dokuya
bakmaya gerek yok.

Merkez başına renk profilleri:

| merkez | R ort | G ort | B ort | R std | G std | B std |
|---|---|---|---|---|---|---|
| 0 | 0,7322 | 0,6008 | 0,6985 | 0,1456 | 0,1751 | 0,1320 |
| 1 | 0,6092 | 0,4604 | 0,5922 | 0,1453 | 0,1684 | 0,1313 |
| 2 | 0,6740 | 0,4852 | 0,7404 | 0,1467 | 0,1657 | 0,1016 |
| 3 | 0,6852 | 0,4888 | 0,6192 | 0,1480 | 0,1592 | 0,1219 |
| 4 | 0,8001 | 0,6716 | 0,8194 | 0,1087 | 0,1295 | 0,0925 |

Merkezler arası en büyük ortalama farkı 0,2272 (mavi kanal). Karışıklık matrisinde
m2 ve m4 neredeyse kusursuz ayrılıyor, **m0 ile m3 sürekli birbirine karışıyor** —
muhtemelen benzer boyama protokolü.

### Açıkça ayrıştıramadığımız şey

Renk uzaklığı ile düşüş arasındaki korelasyon **-0,624**: renk olarak eğitim
setinden en uzak merkez, en az kaybeden merkez. Eğitim setinin iç renk çeşitliliği
ile düşüş arasındaki korelasyon ise **+0,812**.

İkisi birbirinin aynası: beş merkezden üçü eğitime seçildiğinde, seçilenler renk
olarak çeşitliyse geride kalan merkez zorunlu olarak ortalamaya yakın düşer (iki
değişken arası korelasyon -0,643). **Beş nokta ile hangisinin gerçek sebep olduğu
ayrıştırılamaz, ve buradan korelasyonla ilerlemek sebep uydurmak olur.** Doğru
yöntem müdahale deneyi.

---

## Müdahale: rengi bozmak ya da silmek düşüşü kapatıyor mu?

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

## Tasarım kararları ve gerekçeleri

Portfolyo değeri modelde değil, bu kararlarda.

- **Ayrım hasta bazında.** Aynı hastanın karoları hem eğitimde hem testte olursa
  model merkezi değil hastayı ezberler.
- **Model seçimi iç doğrulamada.** Dış kümeye bakarak en iyi turu seçmek, sahada
  elinde olmayan bilgiyi kullanmaktır; hedef hastaneden sızıntıdır.
- **Taban çizgisinde renk düzeltmesi yok.** Renk normalizasyonu bizim tedavimiz.
  Tedaviyi kontrol grubuna koyarsan işe yarayıp yaramadığını ölçemezsin.
- **Normalizasyon yalnız eğitim karolarından.**
- **Her katta aynı eğitim bütçesi.** Yoksa fark merkezden değil veri miktarından gelir.
- **Ölçüm kümeleri eşit:** üç kümede de aynı hasta sayısı, hasta başına aynı karo,
  hasta içinde 50/50 etiket.

### Ölçüm kümelerinin neden bu kadar küçük olduğu

İlk tam koşu 6 hastalık ölçüm kümeleriyle kuruldu ve **kümeler dolmadı**: iç ölçüm
536 karoda kaldı, hasta başına dağılım [8, 20, 52, 78, 78, 300] oldu — tek hasta
kümenin yarısından fazlasını oluşturuyordu.

Sebep: veri setinin %50/%50 tümör dengesi **küresel**, hasta düzeyinde değil.
Hastaların çoğunda tümör dokusu yok denecek kadar az. Her sınıftan en az 150
karosu olan hasta sayısı merkez başına **5 · 5 · 5 · 3 · 3**.

> **"455.954 karo" cümlesi aldatıcıdır. Belirsizlik karolardan değil hastalardan
> gelir, ve elde merkez başına üç ile beş hasta vardır.**

Kümeler bu tavana göre yeniden kuruldu: üç kümede de 3 hasta, 900 karo, hasta
başına eşit 300 karo.

---

## Pilotun yanlış sonucu ve nasıl bulundu

İlk pilot koşusunda **beklenenin tersi** çıktı: dış merkezler iç ölçümden daha iyi
göründü (iç AUC 0,9570, dış test 0,9686).

Sebep model değil ölçüm kurgusuydu: iç doğrulama 6 hastadan 2.612 karo, dış kümeler
8-9 hastadan 30.000 karoydu. Ayrıca iç doğrulama da görülmemiş hastalardan
oluşuyordu, yani orada da kayma vardı — **merkez kaymasıyla hasta kayması
kıyaslanmıştı.** Kendi koyduğumuz "aynı büyüklük" kuralı çiğnenmişti.

Düzeltildikten sonra düşüş 15 koşunun 14'ünde pozitif çıktı.

---

## Veri

**Camelyon17-WILDS.** Meme kanserinin lenf nodu metastazı, H&E boyalı patoloji
kesitlerinden 96x96 karolar. Hollanda'da beş hastane.

| | |
|---|---|
| toplam karo | 455.954 |
| hasta / slayt / merkez | 43 / 50 / 5 |
| tümör oranı | her merkezde tam %50 |
| kullanılan alt küme | 149.997 karo (4,15 GB uint8) |

Bölünme dosyadan doğrulandı: merkez 0, 3, 4 eğitim · merkez 1 dış doğrulama ·
merkez 2 dış test. Tam koşuda bu bölünme beş kat halinde döndürüldü.

⚠️ **Etiket kayması yok, saf kovaryat kayması var.** Tümör oranı beş merkezde de
%50,00 — WILDS bilerek dengelemiş. Deney temiz ama **yapay**: gerçek hastanelerde
sevk deseni farklıdır, yani buradaki düşüş sahadakinin alt sınırıdır.

⚠️ Parquet dosyaları sıralı: 21 dosyanın 6'sı tek merkezli, 8'i tek etiketli.
Dosya sırasından örnek almak tek sınıflı eğitim kümesi üretir. Seçim global
karıştırma + merkez/etiket katmanlı yapıldı.

---

## Hat

| Script | Ne yapar |
|---|---|
| `indir.sh` / `tamamla.sh` | Doğrudan HTTP indirme + boyut doğrulaması |
| `00_kesif.py` | Metadata analizi, bölünme doğrulaması, sızıntı kontrolü |
| `01_altkume.py` | Merkez başına 30 bin karo, etiket dengeli, hasta orantılı |
| `01b_dogrula.py` | Bellek dostu doğrulama (mmap) |
| `02_pilot.py` | Tek kat, hattın çalıştığını görmek için |
| `03_tam.py` | 5 kat x 3 tohum, eşitlenmiş ölçüm kümeleri |
| `04_teshis.py` | Renk istatistikleri, 6 sayıdan ve CNN'den merkez tahmini |
| `05_mudahale.py` | `--kol renk_artirma` / `--kol gri` |

Model: ImageNet ön eğitimli ResNet-18, son katman tek çıkışlı, tüm ağ eğitiliyor.
Aygıt: Apple MPS. Bir koşu ~111 saniye, 15 koşu ~28 dakika.

---

## Sınırlar

- Her ölçüm **3 hasta ve 900 karo** üzerinden. Belirsizlik hastadan geliyor.
- 15 koşu bağımsız değil; katlar aynı veriyi paylaşıyor.
- Düşüşün standart sapması (0,0223) ortalamasıyla (0,0244) neredeyse aynı:
  **yön güvenilir, büyüklük belirsiz.**
- ImageNet ön eğitimi renge duyarlı özellik taşıyor. Ölçülen düşüşün bir kısmı
  ön eğitimden geliyor olabilir.
- Tümör oranı yapay olarak dengelenmiş; gerçek dağılım kaymasını içermiyor.

## Çalıştırma

```bash
uv sync
./indir.sh && ./tamamla.sh
uv run 00_kesif.py
uv run 01_altkume.py && uv run 01b_dogrula.py
uv run 03_tam.py
uv run 04_teshis.py
uv run 05_mudahale.py --kol renk_artirma
uv run 05_mudahale.py --kol gri
```
