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

---

## Üç cümlede sonuç

1. **Düşüş var ama AUC'de görünmüyor.** AUC %2,5 kaybederken kalibrasyon %43
   bozuluyor. Sıralama taşınıyor, eşik taşınmıyor.
2. **Merkez imzası renkte.** Karo başına 6 sayı, hangi hastaneden geldiğini %86
   doğrulukla biliyor — 27.648 pikselin yaptığı işin %91'i.
3. **Ama kayıp renkten gelmiyor.** Rengi bozduk ve tamamen sildik; düşüş kapanmadı.
   **Merkezin renkten ayırt edilebilir olması, kaybın renkten geldiği anlamına gelmez.**

---

## Ayrıntılar

| | |
|---|---|
| [Veri ve sınırlar](docs/01-veri.md) | Camelyon17-WILDS, alt küme seçimi, ölçümün gerçek birim sayısı |
| [Ölçüm](docs/02-olcum.md) | Tasarım kararları ve gerekçeleri, pilotun yanlış sonucu ve nasıl bulunduğu |
| [Teşhis](docs/03-teshis.md) | Renk istatistikleri, 6 sayıdan ve CNN'den merkez tahmini |
| [Müdahale](docs/04-mudahale.md) | Renk artırma ve gri tonlama kolları, kat bazlı sonuçlar |

---

## Yapı

```
scriptler/   00_kesif -> 06_karsilastir, sirayla calisir
docs/        ayrintili notlar
sonuclar/    csv ciktilari
loglar/      kosu ciktilari
izle.sh      arka planda kosan mudahaleyi canli izler
```

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
| `06_karsilastir.py` | Üç kolu yan yana koyar |

Model: ImageNet ön eğitimli ResNet-18, son katman tek çıkışlı, tüm ağ eğitiliyor.
Aygıt: Apple MPS. Bir koşu ~111 saniye, 15 koşu ~28 dakika.

## Çalıştırma

```bash
uv sync
./scriptler/indir.sh && ./scriptler/tamamla.sh
uv run scriptler/00_kesif.py
uv run scriptler/01_altkume.py && uv run scriptler/01b_dogrula.py
uv run scriptler/03_tam.py
uv run scriptler/04_teshis.py
uv run scriptler/05_mudahale.py --kol renk_artirma
uv run scriptler/05_mudahale.py --kol gri
uv run scriptler/06_karsilastir.py
```

Scriptler kök dizinden çalıştırılır; veri yollarıyla çıktı yolları buna göre kurulu.
