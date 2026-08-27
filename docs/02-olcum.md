[← README](../README.md)

# Ölçüm: tasarım kararları ve pilotun düzeltilmesi

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
