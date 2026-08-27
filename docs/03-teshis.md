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


---

## İkinci tur: renk silinince imza kalıyor mu? (`07_teshis2.py`)

Blok 6 rengi bozma ve silme müdahalelerinin düşüşü kapatmadığını gösterdi. Buradan
"kayıp renkten gelmiyor" sonucu çıkarmak cazipti. Önce şunu test etmek gerekiyordu:
**müdahale imzayı gerçekten sildi mi?**

| yöntem | merkez tahmin doğruluğu |
|---|---|
| şans seviyesi | 0,2000 |
| keskinlik 2 sayı (Laplacian varyansı + yüksek frekans) | **0,2038** |
| gri 2 sayı (luma ortalama + std) | 0,3319 |
| dört sayı birlikte | 0,4241 |
| **CNN, gri görüntü** | **0,7903** |
| renk 6 sayı | 0,8605 |
| CNN, renkli görüntü | 0,9445 |

### Bulgu 1: renk silinse bile merkez tanınıyor

CNN gri görüntüde %79 doğruluk veriyor: şansın dört katı, renkli CNN'in %84'ü.
**Merkez imzası yalnızca boyada değil, dokunun kendisinde de var.**

Bu, Blok 6 hükmünü düzeltir:

> Rengi silmek merkez imzasını silmiyor, %94'ten %79'a indiriyor. Müdahale imzanın
> kaynağını ortadan kaldırmadığı için kaybın kapanmaması sürpriz değil.
> **"Kayıp renkten gelmiyor" demek eldeki veriyle fazla güçlü bir iddia.**

Doğru ifade: merkez imzası hem renkte hem dokuda; sadece rengi silmek yetmiyor.

### Bulgu 2: renk imzası ortalamalarda, doku imzası desende

Renkli görüntüde 6 sayı CNN'in %91'ini yakalıyordu. Gri görüntüde 2 sayı CNN'in
ancak %42'sini yakalıyor (0,3319 / 0,7903). Boya farkı ortalamalara yansıyan basit
bir kayma; doku farkı ancak deseni görebilen bir modelin yakalayabileceği bir şey.

### Bulgu 3: keskinlik merkez ortalamasında ayırıyor, karo bazında ayırmıyor

| merkez | luma ort | luma std | lap var | yüksek frek |
|---|---|---|---|---|
| 0 | 0,6512 | 0,1601 | 0,03129 | 0,12483 |
| 1 | 0,5199 | 0,1564 | 0,03882 | 0,13681 |
| 2 | 0,5707 | 0,1476 | 0,04317 | 0,14735 |
| 3 | 0,5624 | 0,1498 | **0,02364** | 0,10721 |
| 4 | 0,7269 | 0,1165 | **0,05577** | 0,16129 |

En keskin merkez (m4) ile en yumuşak merkez (m3) arasında **2,36 kat** fark var.
Buna rağmen keskinlikten merkez tahmini şans seviyesinde (0,2038): merkez içindeki
karodan karoya değişim, merkezler arası farktan büyük. **Ortalamada görünen fark,
tek örnekte kayboluyor.**

### Gri CNN karışıklık matrisi

```
     m0   m1    m2   m3   m4
m0  568   60     3  179    1
m1   31  769    14   77    0
m2  146  102  1636    8   55
m3   96  217    44  278    2
m4    1    1    30    2  778
```

m4 renk atılsa bile %96 tanınıyor, ve zaten en keskin merkez o. m3'ün doğruluğu
renkli CNN'de 498/637 iken gri CNN'de 278/637'ye düşüyor: **m0 ile m3 ayrımı büyük
ölçüde renkten geliyormuş.**
