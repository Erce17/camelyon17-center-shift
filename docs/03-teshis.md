[← README](../README.md)

# Teşhis: kayma nereden geliyor?

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
