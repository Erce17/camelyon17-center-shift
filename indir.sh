#!/bin/bash
# Camelyon17-WILDS indirici.
# huggingface_hub (xet ile de xetsiz de) dosyalari olusturup tek bayt yazmadi.
# Kanitlanmis yol: dogrudan HTTP. -C - kaldigi yerden devam ettirir.
KOK="https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/resolve/main"
indir() {
  hedef="veri/$1"
  curl -sL -C - --retry 5 --retry-delay 3 -o "$hedef" "$KOK/$1" \
    && echo "TAMAM $1 ($(du -m "$hedef" | cut -f1) MB)" \
    || echo "HATA $1"
}
export -f indir; export KOK
xargs -P 3 -I{} bash -c 'indir "$@"' _ {} < dosyalar.txt
echo "HEPSI BITTI"
