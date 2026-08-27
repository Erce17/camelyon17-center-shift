"""
Camelyon17-WILDS indirme. Kaynak: Hugging Face aynasi.

Resmi kaynak (CodaLab) 26.08.2026 itibariyla HTTP 500 veriyor.
Ayna ayni surumu tasiyor: 455.954 karo, parquet, metadata sutunlari tam.

snapshot_download secildi cunku kaldigi yerden devam eder;
kopan indirmede bastan baslamak 10 GB'ta kabul edilemez.
"""
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"   # xet katmani 1.28de sessizce asili kaliyor

from huggingface_hub import snapshot_download

yol = snapshot_download(
    repo_id="wltjr1007/Camelyon17-WILDS",
    repo_type="dataset",
    local_dir="veri/camelyon17-hf",
    max_workers=6,
    allow_patterns=["data/*.parquet", "README.md"],
)
print("BITTI:", yol)
