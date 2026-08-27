[← README](../README.md)

# Veri ve sınırlar

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

---

## Sınırlar

- Her ölçüm **3 hasta ve 900 karo** üzerinden. Belirsizlik hastadan geliyor.
- 15 koşu bağımsız değil; katlar aynı veriyi paylaşıyor.
- Düşüşün standart sapması (0,0223) ortalamasıyla (0,0244) neredeyse aynı:
  **yön güvenilir, büyüklük belirsiz.**
- ImageNet ön eğitimi renge duyarlı özellik taşıyor. Ölçülen düşüşün bir kısmı
  ön eğitimden geliyor olabilir.
- Tümör oranı yapay olarak dengelenmiş; gerçek dağılım kaymasını içermiyor.
