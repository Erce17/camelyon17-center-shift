#!/bin/bash
# Mudahale kosularini taban cizgisiyle yan yana izler. Cikmak icin Ctrl+C.
cd "$(dirname "$0")" || exit 1
while true; do
  clear
  printf "CAMELYON MUDAHALE IZLEME   %s\n" "$(date '+%H:%M:%S')"
  printf "=========================================================\n\n"

  for kol in renk mudahale_gri; do
    [ "$kol" = "renk" ] && log="mudahale_renk.log" ad="RENK ARTIRMA" || { log="mudahale_gri.log"; ad="GRI TONLAMA"; }
    if [ -f "$log" ]; then
      bitti=$(grep -c "bitti" "$log")
      son=$(grep "toplam" "$log" | tail -1 | grep -oE "toplam [0-9.]+ dk")
      printf "%-14s  %2s/15 kosu   %s\n" "$ad" "$bitti" "$son"
      grep "DUSUS" "$log" | tail -3 | sed 's/^  -> /                /'
      if grep -q "GENEL DUSUS" "$log"; then
        printf "  >> %s\n" "$(grep 'GENEL DUSUS' "$log")"
      fi
    else
      printf "%-14s  henuz baslamadi\n" "$ad"
    fi
    printf "\n"
  done

  printf "TABAN CIZGISI (03_tam.py, bugun kosuldu)\n"
  printf "  GENEL DUSUS: +0.0244 +- 0.0223 AUC   (15 kosunun 14'unde pozitif)\n"
  printf "  kat basina: m0 +0.0389  m1 +0.0341  m2 +0.0265  m3 +0.0209  m4 +0.0016\n\n"

  printf "GPU: "
  ps aux | grep "[0]5_mudahale.py" >/dev/null && printf "calisiyor\n" || printf "kosu bitti\n"
  printf "\n(Ctrl+C ile cik, kosuyu durdurmaz)\n"
  sleep 5
done
