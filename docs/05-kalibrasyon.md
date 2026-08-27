[← README](../README.md)

# Yeniden kalibrasyon: kaybın ne kadarı ucuza geri gelir?

Ölçüm şunu bırakmıştı: dış merkezde AUC %2,5 kaybediliyor ama Brier %43 bozuluyor.
Sıralama taşınıyor, eşik ve kalibrasyon taşınmıyor.

Bunun test edilebilir bir sonucu var: **hedef hastaneden az sayıda etiketli örnek
alıp yalnızca eşiği/kalibrasyonu yeniden ayarlarsak kaybın ne kadarı geri gelir?**

Model **yeniden eğitilmez.** Skorların üstüne bir dönüşüm takılır, o kadar.
(`08_kalibrasyon.py`, 5 kat x 3 tohum, 28 dakika)

## Sızıntı kuralı

Kalibrasyon karoları, değerlendirme karolarıyla **aynı hastadan olamaz.**
Kalibrasyon seti dış merkezin ölçüm kümesine girmeyen hastalarından çekilir.
Aksi halde "yeni hastaneye uyarladım" derken test hastası ezberlenir.

## Sonuç

```
ic tavan (dogruluk)      : 0.9104
dis taban (mudahalesiz)  : 0.8461
KAPATILACAK ACIK         : +0.0644

ic brier 0.0873 | dis brier 0.1248 | brier acigi +0.0375
```

| yöntem | N | doğruluk | kapanan açık | brier | brier kapanan |
|---|---|---|---|---|---|
| eşik_dış | 20 | 0,8541 | %12,4 | 0,1248 | %0,0 |
| eşik_dış | 50 | 0,8696 | %36,6 | 0,1248 | %0,0 |
| eşik_dış | 100 | 0,8734 | %42,5 | 0,1248 | %0,0 |
| eşik_dış | 200 | 0,8793 | **%51,6** | 0,1248 | %0,0 |
| platt | 20 | 0,8541 | %12,4 | 0,1432 | **%-49,2** |
| platt | 50 | 0,8696 | %36,6 | 0,1218 | %8,0 |
| platt | 100 | 0,8734 | %42,5 | 0,1146 | %27,0 |
| platt | 200 | 0,8793 | **%51,6** | 0,1108 | **%37,3** |

**50 etiketli karo doğruluk kaybının %37'sini, 200 karo %52'sini geri getiriyor.**
Kazancın büyük kısmı ilk elli örnekte geliyor; sonrası hızla yavaşlıyor.

### Üç pratik kural

**1. Eşik taşımak ile olasılık düzeltmek farklı işler.** `eşik_dış` Brier'i hiç
değiştirmiyor (%0): eşik oynatmak kararı düzeltir, olasılıkları düzeltmez.
Güvenilir olasılık isteyen Platt'a ihtiyaç duyar.

**2. Az örnekle Platt zarar verir.** N=20'de Brier %49 **kötüleşiyor** — 20 noktaya
lojistik eğri oturtmak aşırı uyumdur. Kural: **100'ün altında olasılık kalibrasyonu
yapma, sadece eşiği taşı.**

**3. Ek verinin getirisi hızla doyuyor.** Eşik tek bir sayıdır; yeterince örnekle
doğru değeri bulduktan sonra ek veri yalnızca tahminin gürültüsünü azaltır. Açığın
kapanmayan yarısı veri azlığından değil, **sıralamanın kendisinde kaybolmuş
olmasından** kaynaklanır — o kısım eşik ayarıyla geri gelmez.

## Kat bazında (platt, N=200)

| kat | tavan | taban | platt | kapanan |
|---|---|---|---|---|
| 0 | 0,8922 | 0,8259 | 0,8433 | %26,3 |
| 1 | 0,9159 | 0,8044 | 0,8663 | %55,5 |
| 2 | 0,9093 | 0,8341 | 0,8744 | %53,7 |
| 3 | 0,9330 | **0,9185** | 0,9207 | %15,4 |
| 4 | 0,9019 | 0,8474 | 0,8915 | **%81,0** |

Kat 3'ün düşük yüzdesi yanıltıcı: oradaki açık zaten 0,0145, kapatacak bir şey yok.

## ⚠️ N=400 neden raporlanmıyor

İlk çıktıda N=400 satırı kapanma oranını %23,7'ye düşürüyor gibi görünüyordu.
Sebep kalibrasyon değil örneklem: **N=400 yalnızca 15 koşunun 9'unda üretilebildi.**
Kat 3 ve kat 4'te dış merkezin ölçüm dışı hastalarında 400 dengeli karo yok, havuz
yetmiyor. Kod bu durumda havuzun tamamını alıp kısmi N (266, 286) olarak kaydetmiş.

15 koşunun tamamında geçerli olan N değerleri **20, 50, 100, 200**; karşılaştırma
yalnızca bunlarla yapılır.

## Bu deneyin ürün karşılığı

> Yeni bir hastaneye kurulumda 50 ila 200 etiketli örnek istenir. Model yeniden
> eğitilmeden, yalnızca karar eşiği o merkeze taşınarak doğruluk kaybının yarısı
> geri alınır. Kalan yarısı için gerçek uyarlama gerekir.

**Sıradaki test:** etiketsiz uyarlama. Hedef merkezin **etiketsiz** karolarıyla
BatchNorm istatistiklerini güncellemek, açığın kapanmayan yarısına dokunuyor mu?
Etiket gerektirmediği için sahada en ucuz müdahale odur.
