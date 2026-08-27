#!/bin/bash
# Eksik dosyalari BASTAN indirir. Range istegi bu CDN'de 100 kat yavas
# (olculdu: taze 6.4 MB/s, range 62 KB/s), o yuzden devam ettirmiyoruz.
KOK="https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS/resolve/main"
eksikler=$(while read ad boyut; do
  g=$(stat -f %z "veri/data/$ad" 2>/dev/null || echo 0)
  [ "$g" != "$boyut" ] && echo "$ad"
done < beklenen.txt)
cek() {
  ad=$1; bek=$(grep "^$ad " beklenen.txt | awk '{print $2}')
  for d in 1 2 3; do
    rm -f "veri/data/$ad"
    curl -sL --retry 3 -o "veri/data/$ad" "$KOK/data/$ad"
    g=$(stat -f %z "veri/data/$ad" 2>/dev/null || echo 0)
    [ "$g" = "$bek" ] && { echo "TAMAM $ad"; return; }
    echo "eksik kaldi ($g/$bek), deneme $d: $ad"
  done
  echo "HALA EKSIK $ad"
}
export -f cek; export KOK
echo "$eksikler" | xargs -P 3 -I{} bash -c 'cek "$@"' _ {}
echo "HEPSI BITTI"
