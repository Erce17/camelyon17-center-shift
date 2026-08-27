"""
SIZINTI TESTLERI.

Bu depo boyunca uc kural elle korundu. Elle korunan kural, kod buyudukce bozulur.
Bu testler onlari otomatiklestirir:

  1. Ayrim HASTA bazinda. Ayni hastanin karolari hem egitimde hem olcumde olamaz.
  2. Uyarlama/kalibrasyon karolari, degerlendirme hastalarindan gelemez.
  3. Olcum kumeleri esit: ayni hasta sayisi, hasta basina ayni karo, 50/50 etiket.

Calistirma:  uv run pytest testler/ -v
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HASTA_BASI_SINIF = 150
KOK = Path(__file__).resolve().parent.parent


def manifestler():
    return sorted((KOK / "modeller").glob("*.json"))


def veri():
    yol = KOK / "veri/altkume/secim.parquet"
    if not yol.exists():
        pytest.skip("alt kume yok; once 01_altkume.py calistirilmali")
    return pd.read_parquet(yol)


# --------------------------------------------------------------- 1. hasta ayrimi
@pytest.mark.parametrize("mf", manifestler(), ids=lambda p: p.stem)
def test_olcum_kumeleri_hasta_bazinda_ayrik(mf):
    """Ic dogrulama, dis dogrulama ve dis test hastalari birbiriyle kesismemeli."""
    b = json.loads(mf.read_text())["bolunme"]
    ic = set(b["ic_dogrulama_hastalari"])
    dd = set(b["dis_dogrulama_hastalari"])
    dt = set(b["dis_test_hastalari"])
    assert not (ic & dd), f"ic dogrulama ile dis dogrulama kesisiyor: {ic & dd}"
    assert not (ic & dt), f"ic dogrulama ile dis test kesisiyor: {ic & dt}"
    assert not (dd & dt), f"dis dogrulama ile dis test kesisiyor: {dd & dt}"


@pytest.mark.parametrize("mf", manifestler(), ids=lambda p: p.stem)
def test_dis_merkezler_egitimde_kullanilmamis(mf):
    """Dis dogrulama ve dis test merkezleri egitim merkezlerinden olamaz."""
    b = json.loads(mf.read_text())["bolunme"]
    assert b["dis_dogrulama_merkezi"] not in b["egitim_merkezleri"]
    assert b["dis_test_merkezi"] not in b["egitim_merkezleri"]
    assert len(b["egitim_merkezleri"]) == 3


@pytest.mark.parametrize("mf", manifestler(), ids=lambda p: p.stem)
def test_ic_dogrulama_hastalari_egitim_merkezlerinden(mf):
    """Ic dogrulama, egitim merkezlerinin GORULMEYEN hastalarindan olusur."""
    md = veri()
    b = json.loads(mf.read_text())["bolunme"]
    for h in b["ic_dogrulama_hastalari"]:
        merkezler = set(md[md.patient == h].center.unique())
        assert merkezler <= set(b["egitim_merkezleri"]), \
            f"hasta {h} egitim merkezlerinde degil: {merkezler}"


# --------------------------------------------------------- 2. olcum kumesi esitligi
@pytest.mark.parametrize("mf", manifestler(), ids=lambda p: p.stem)
def test_olcum_kumeleri_esit_buyuklukte(mf):
    b = json.loads(mf.read_text())["bolunme"]
    assert b["ic_dogrulama_karo"] == b["dis_test_karo"], \
        "olcum kumeleri esit buyuklukte olmali (pilotun hatasi buydu)"


@pytest.mark.parametrize("mf", manifestler(), ids=lambda p: p.stem)
def test_olcum_kumeleri_etiket_dengeli(mf):
    """Hasta icinde 50/50 kuruldugu icin kume genelinde de 50/50 olmali."""
    md = veri()
    b = json.loads(mf.read_text())["bolunme"]
    for anahtar, merkez in [("dis_test_hastalari", b["dis_test_merkezi"]),
                            ("dis_dogrulama_hastalari", b["dis_dogrulama_merkezi"])]:
        d = md[(md.center == merkez) & (md.patient.isin(b[anahtar]))]
        assert len(d) > 0
        # her hastanin her siniftan yeterli karosu olmali
        for h in b[anahtar]:
            g = d[d.patient == h]
            assert min((g.label == 1).sum(), (g.label == 0).sum()) >= HASTA_BASI_SINIF, \
                f"hasta {h} olcume uygun degil (her siniftan >={HASTA_BASI_SINIF} karo gerekir)"


# ------------------------------------------------------- 3. uyarlama ciktisi ayrik
def test_uyarlama_ciktisi_sizintisiz():
    """uyarla.py: uyarlama hastalari ile degerlendirme hastalari kesismemeli."""
    yol = KOK / "sonuclar/uyarlama_merkez0.csv"
    if not yol.exists():
        pytest.skip("uyarla.py henuz calistirilmamis")
    d = pd.read_csv(yol)
    assert "rol" in d.columns, "cikti 'rol' sutunu tasimali (uyarla.py guncel mi?)"
    deg = set(d[d.rol == "degerlendirme"].hasta)
    uya = set(d[d.rol == "uyarlama"].hasta)
    assert deg, "degerlendirme kumesi bos"
    assert uya, "uyarlama kumesi bos"
    assert not (deg & uya), f"SIZINTI: ayni hasta hem uyarlamada hem degerlendirmede {deg & uya}"
    assert set(d[d.rol == "degerlendirme"].etiket.unique()) == {0, 1}


def test_uyarlama_kumesi_yeterli_buyuklukte():
    """400 karonun altinda BN uyarlamasi zarar veriyor (docs/06)."""
    yol = KOK / "sonuclar/uyarlama_merkez0.csv"
    if not yol.exists():
        pytest.skip("uyarla.py henuz calistirilmamis")
    d = pd.read_csv(yol)
    n = int((d.rol == "uyarlama").sum())
    assert n >= 400, f"uyarlama kumesi {n} karo; 400 altinda BN uyarlamasi zarar verir"


# ----------------------------------------------------------- 4. veri butunlugu
def test_hasta_tek_merkeze_ait():
    """Bir hasta birden fazla merkezde gorunemez; gorunurse hasta bazli ayrim coker."""
    md = veri()
    coklu = md.groupby("patient").center.nunique()
    assert (coklu == 1).all(), f"birden fazla merkezde gorunen hasta: {coklu[coklu>1].index.tolist()}"


def test_altkume_etiket_dengeli():
    md = veri()
    for m, g in md.groupby("center"):
        oran = g.label.mean()
        assert 0.45 <= oran <= 0.55, f"merkez {m} etiket orani {oran:.3f}"
