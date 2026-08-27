[← README](../README.md)

# Etiketsiz uyarlama: BatchNorm istatistiklerini hedef merkeze taşımak

[Yeniden kalibrasyon](05-kalibrasyon.md) şunu bırakmıştı: doğruluk kaybının yarısı
sadece eşik taşıyarak geri geliyor, kalan yarısı sıralamanın kendisinde kaybolmuş.

Bu deney o kalan yarıya dokunmayı dener, ve **hiç etiket kullanmadan.**
(`09_bn_uyarlama.py`, 5 kat x 3 tohum, 30 dakika)

## Fikir

ResNet-18'in içinde **20 BatchNorm katmanı** var. Her biri kendinden önceki katmanın
çıktısını normalize eder, ve kullandığı ortalama/varyans **eğitim verisinin**
istatistikleridir.

Model dış merkeze götürüldüğünde içeride hâlâ eğitim hastanelerinin dağılımına göre
normalize eder: yanlış zemin, ve hata her katmanda büyüyerek ilerler.

Çözüm: hedef merkezin **etiketsiz** karolarını modelden bir kez geçir, BN'lerin
biriktirdiği istatistikleri o merkeze göre yeniden hesapla. **Ağırlıklara dokunma.**

```python
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        m.reset_running_stats()
        m.momentum = None          # kumulatif ortalama
model.train()
with torch.no_grad():              # gradyan yok, agirlik guncellenmiyor
    for x, _ in hedef_veri:        # etiket hic kullanilmiyor
        model(x)
model.eval()
```

Ortalama ve varyans hesaplamak için görüntünün tümörlü olup olmadığını bilmek
gerekmez. Sahada bunun anlamı büyük: **etiketli veri patoloğun zamanı demektir,
ham görüntü ise zaten arşivde durur.**

Bu veride işleme şansının yüksek olmasının sebebi: tümör oranı beş merkezde de %50,
yani **saf kovaryat kayması** var, etiket kayması yok.

## Sonuç

```
      metrik  ic (tavan)  dis taban      acik
         AUC      0.9726     0.9482    0.0244
    dogruluk      0.8896     0.8461    0.0435
   duy@ozg90      0.9218     0.8579    0.0639
       brier      0.0873     0.1248    0.0375
```

Açığın yüzde kaçı kapandı:

| kol | AUC | doğruluk | duy@özg90 | brier | etiket |
|---|---|---|---|---|---|
| eşik200 | %0,0 | %76,3 | %0,0 | %0,0 | 200 adet |
| bn100 | **%-118,3** | %-24,0 | %-116,5 | **%-311,3** | YOK |
| **bn_tam** | **%67,0** | %29,5 | **%76,3** | **%98,7** | **YOK** |
| bn_tam+eşik | %67,0 | %119,3 | %76,3 | %98,7 | 200 adet |

**Etiketsiz uyarlama, etiketli müdahaleden daha fazla iş görüyor.** Tek bir etiket
almadan, ~800 ham karoyla:

- AUC açığının **%67'si** kapandı — eşik ayarı buna hiç dokunamıyordu (%0)
- Duyarlılık açığının **%76'sı**
- **Kalibrasyon açığının %98,7'si.** Brier 0,1248 → 0,0878; iç doğrulamadaki 0,0873
  ile pratikte aynı. Modelin olasılıkları yeni hastanede yeniden güvenilir.

`bn_tam+eşik` doğruluk açığının %119'unu kapatıyor, yani iç tavanın üstüne çıkıyor.
Bunu "model dış merkezde daha iyi" diye okumak yanlış olur: iç doğrulama kümesinin
de kendi zorluğu var ve 3 hastalık ölçümlerde bu sapma normal.

## Kalibrasyon hatasının yönü

Brier'in bozulması "model yanlış" der ama yönünü söylemez. Reliability diagram
söyler:

```
            ic dogrulama: ort tahmin 0.3264 | gercek oran 0.5000
   dis test, uyarlamasiz: ort tahmin 0.2738 | gercek oran 0.5000
    dis test, BN sonrasi: ort tahmin 0.4727 | gercek oran 0.5000

TUMOR karolarina verilen ortalama skor:
   uyarlamasiz: tumor 0.5290 | temiz 0.0185
    BN sonrasi: tumor 0.9029 | temiz 0.0425
```

Model dış merkezde **gereğinden düşük** olasılık veriyor. Tümör karolarına verdiği
ortalama skor 0,53'te kalıyor: tümörü görüyor ama emin olamıyor, ve eşiğin altında
kalanları kaçırıyor. **Fazla emin biçimde yanılmıyor, fazla temkinli davranıp
tümörü kaçırıyor** — duyarlılık düşüşünün doğrudan sebebi bu.

Etiketsiz uyarlama ortalama tahmini 0,4727'ye (gerçek oran 0,50), tümör skorunu
0,90'a taşıyor. Grafik: [`gorseller/01_kalibrasyon.png`](../gorseller/01_kalibrasyon.png)

## ⚠️ Üç uyarı, sonucun kendisi kadar önemli

**1. Az veriyle felaket.** `bn100` her metrikte tabandan kötü; Brier %311
kötüleşiyor. 100 karo, 20 BN katmanının ortalama ve varyansını kestirmek için
yetersiz. **Gürültülü istatistik, sağlam olandan beterdir.**
Aynı desen Platt kalibrasyonunda N=20'de de görülmüştü.

**2. Tutarlı değil.** `bn_tam` 15 koşunun 10'unda AUC'yi artırdı, 5'inde düşürdü.
Ortalama +0,0163, standart sapma 0,0322 — sapma ortalamanın iki katı.

| kat | bn_tam AUC değişimi |
|---|---|
| 0 | +0,0622 |
| 1 | **-0,0187** |
| 2 | +0,0230 |
| 3 | **-0,0079** |
| 4 | +0,0231 |

**3. Ölçüm hâlâ 3 hasta ve 900 karo üzerinden.** Depo boyunca geçerli olan sınır
burada da geçerli.

## Dürüst ifade

> Etiketsiz BatchNorm uyarlaması bu veride ortalama olarak güçlü bir kazanç sağlıyor,
> özellikle kalibrasyonda neredeyse tam düzeltme yapıyor. Ama yeterli veri ister ve
> her merkezde işe yaramaz.

## Uyarlama merdiveni

Bu deponun ölçtüğü müdahaleler, ucuzdan pahalıya:

| seviye | ne gerekir | ne kazandırır |
|---|---|---|
| eşik taşıma | 50-200 etiketli örnek | doğruluk açığının %37-52'si |
| **BN uyarlaması** | **etiketsiz ham görüntü** | **AUC %67, kalibrasyon %99** |
| ikisi birlikte | ikisi | doğruluk açığının tamamı |
| kısmi ince ayar | birkaç yüz etiketli örnek | ölçülmedi |
| tam yeniden eğitim | çok sayıda etiketli örnek | ölçülmedi |

**En ucuz müdahale en çok getiriyi sağlıyor** — yeterli ham veri verildiği sürece.
