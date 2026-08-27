"""
9. BLOK: MODEL KAYDI VE MANIFEST.

Bu depoda simdiye kadar hicbir agirlik kaydedilmedi: her script modeli sifirdan
egitip olcup atti. Olcum icin dogru, urun icin degil. Bu script kalici artefakt
uretir:

  modeller/kat<K>_tohum<T>.pt        agirliklar
  modeller/kat<K>_tohum<T>.json      manifest
  sonuclar/skorlar_kat<K>_tohum<T>.csv   karo bazli skorlar (grafikler icin)

MANIFEST neden onemli: uc ay sonra gelen "bu model hangi veriyle egitildi" sorusunun
cevabi budur. Urun belli bir veri kumesiyle dogrulandigi icin regulasyon da bunu
ister. Ayni disiplin dicom-data-pipeline deposunda uygulanmisti; burada eksikti.

Varsayilan tek kat/tohum kaydeder (hizli). --hepsi ile 15'inin tamami.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from torchvision.models import resnet18, ResNet18_Weights

MERKEZLER = [0, 1, 2, 3, 4]
MERKEZ_BASI = 10_000
IC_DOGRULAMA_HASTA = 1
OLCUM_HASTA = 3
HASTA_BASI_SINIF = 150
TUR = 5
YIGIN = 128

ap = argparse.ArgumentParser()
ap.add_argument("--kat", type=int, default=0)
ap.add_argument("--tohum", type=int, default=42)
ap.add_argument("--hepsi", action="store_true", help="15 kat/tohum kombinasyonunun tamami")
args = ap.parse_args()

aygit = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
Path("modeller").mkdir(exist_ok=True)

X = np.load("veri/altkume/X.npy", mmap_mode="r")
md = pd.read_parquet("veri/altkume/secim.parquet").reset_index(drop=True)
md["idx"] = np.arange(len(md))
etiketler = md.label.values.astype(np.float32)

_h = md.groupby(["center", "patient"]).label.agg(["size", "sum"])
_h["min_sinif"] = np.minimum(_h["sum"], _h["size"] - _h["sum"])
UYGUN = {m: np.sort(d[d.min_sinif >= HASTA_BASI_SINIF].index.get_level_values(1).values)
         for m, d in _h.groupby(level=0)}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def veri_ozeti():
    """Alt kumenin kimligi: sekil + ilk/son bloklarin sha256'si."""
    h = hashlib.sha256()
    h.update(str(X.shape).encode())
    h.update(np.asarray(X[:200]).tobytes())
    h.update(np.asarray(X[-200:]).tobytes())
    return h.hexdigest()


def olcum_kumesi(kaynak, hastalar, tohum):
    parcalar = []
    for hh in hastalar:
        g = kaynak[kaynak.patient == hh]
        t, n = g[g.label == 1], g[g.label == 0]
        k = min(HASTA_BASI_SINIF, len(t), len(n))
        if k == 0:
            continue
        parcalar.append(t.sample(n=k, random_state=tohum))
        parcalar.append(n.sample(n=k, random_state=tohum))
    return pd.concat(parcalar)


def olc(skor, y, esik=0.5):
    fpr, tpr, _ = roc_curve(y, skor)
    return dict(auc=roc_auc_score(y, skor),
                dogruluk=float(((skor >= esik).astype(int) == y).mean()),
                duyarlilik_ozg90=float(np.interp(0.10, fpr, tpr)),
                brier=brier_score_loss(y, skor))


def egit_kaydet(kat, tohum):
    t0 = time.time()
    dis_test, dis_dog = kat, (kat + 1) % 5
    egitim_merkez = [m for m in MERKEZLER if m not in (dis_test, dis_dog)]
    torch.manual_seed(tohum); np.random.seed(tohum)
    rng = np.random.default_rng(tohum)

    ic_hastalar = []
    for m in egitim_merkez:
        ic_hastalar += list(rng.choice(UYGUN[m], IC_DOGRULAMA_HASTA, replace=False))
    ic_dog = olcum_kumesi(md[md.center.isin(egitim_merkez) & md.patient.isin(ic_hastalar)],
                          ic_hastalar, tohum)
    dis = {}
    for ad, m in [("dis_dogrulama", dis_dog), ("dis_test", dis_test)]:
        sec = rng.choice(UYGUN[m], min(OLCUM_HASTA, len(UYGUN[m])), replace=False)
        dis[ad] = (olcum_kumesi(md[md.center == m], sec, tohum), sec)

    havuz = md[md.center.isin(egitim_merkez) & ~md.patient.isin(ic_hastalar)]
    parcalar = []
    for m in egitim_merkez:
        g = havuz[havuz.center == m]
        for e in (0, 1):
            alt = g[g.label == e]
            parcalar.append(alt.sample(n=min(MERKEZ_BASI // 2, len(alt)), random_state=tohum))
    egitim = pd.concat(parcalar)

    ornek = np.asarray(X[np.sort(egitim.idx.values)[:5000]]).astype(np.float32) / 255.0
    ort, std = ornek.mean(axis=(0, 1, 2)), ornek.std(axis=(0, 1, 2))
    ort_t = torch.tensor(ort, device=aygit).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=aygit).view(1, 3, 1, 1)

    def yigin(idxler, karistir, artir):
        sira = np.random.permutation(len(idxler)) if karistir else np.arange(len(idxler))
        for i in range(0, len(sira), YIGIN):
            alt = np.sort(idxler[sira[i:i + YIGIN]])
            x = torch.from_numpy(np.asarray(X[alt]).copy()).to(aygit)
            x = x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            if artir:
                if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[3])
                if torch.rand(1).item() < 0.5: x = torch.flip(x, dims=[2])
            yield (x - ort_t) / std_t, alt

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 1)
    model = model.to(aygit)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    kayip_fn = nn.BCEWithLogitsLoss()

    def skorla(d):
        model.eval(); skor = np.zeros(len(d), dtype=np.float32)
        yer = {v: i for i, v in enumerate(d.idx.values)}
        with torch.no_grad():
            for x, alt in yigin(d.idx.values, False, False):
                p = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
                for j, a in enumerate(alt): skor[yer[a]] = p[j]
        return skor, d.label.values

    en_iyi, en_iyi_durum, tur_gecmisi = -1, None, []
    for tur in range(1, TUR + 1):
        model.train(); toplam, n = 0.0, 0
        for x, alt in yigin(egitim.idx.values, True, True):
            y = torch.from_numpy(etiketler[alt]).to(aygit)
            opt.zero_grad(); k = kayip_fn(model(x).squeeze(1), y)
            k.backward(); opt.step()
            toplam += k.item() * len(alt); n += len(alt)
        s, y = skorla(ic_dog); m = olc(s, y)
        tur_gecmisi.append(dict(tur=tur, kayip=toplam / n, ic_auc=m["auc"]))
        print(f"  tur {tur}: kayip {toplam/n:.4f} | ic AUC {m['auc']:.4f}", flush=True)
        if m["auc"] > en_iyi:
            en_iyi, secilen_tur = m["auc"], tur
            en_iyi_durum = {kk: v.detach().clone() for kk, v in model.state_dict().items()}

    model.load_state_dict(en_iyi_durum)

    # --- karo bazli skorlar: grafikler ve sonraki analizler icin ---
    kayitlar, olcumler, esik_ic = [], {}, None
    for ad, d in [("ic_dogrulama", ic_dog), ("dis_dogrulama", dis["dis_dogrulama"][0]),
                  ("dis_test", dis["dis_test"][0])]:
        s, y = skorla(d)
        if ad == "ic_dogrulama":
            # Karar esigi IC dogrulamada secilir; sahada hedef merkezin etiketi yoktur.
            neg = np.sort(s[y == 0])
            esik_ic = float(neg[min(int(len(neg) * 0.90), len(neg) - 1)])
        olcumler[ad] = olc(s, y, esik_ic if esik_ic is not None else 0.5)
        kayitlar.append(pd.DataFrame(dict(kume=ad, idx=d.idx.values, merkez=d.center.values,
                                          hasta=d.patient.values, etiket=y, skor=s)))
    skorlar = pd.concat(kayitlar)
    ad_kok = f"kat{kat}_tohum{tohum}"
    skorlar.to_csv(f"sonuclar/skorlar_{ad_kok}.csv", index=False)

    torch.save({"state_dict": en_iyi_durum, "mimari": "resnet18",
                "cikis": 1, "normalizasyon": {"ort": ort.tolist(), "std": std.tolist()}},
               f"modeller/{ad_kok}.pt")

    manifest = {
        "ad": ad_kok,
        "olusturuldu": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "veri": {"kaynak": "Camelyon17-WILDS (ayna: wltjr1007/Camelyon17-WILDS)",
                 "altkume_sekli": list(X.shape), "altkume_ozeti_sha256": veri_ozeti(),
                 "toplam_karo": int(len(md))},
        "bolunme": {"kat": kat, "tohum": tohum, "egitim_merkezleri": egitim_merkez,
                    "dis_dogrulama_merkezi": dis_dog, "dis_test_merkezi": dis_test,
                    "ic_dogrulama_hastalari": [int(h) for h in ic_hastalar],
                    "dis_dogrulama_hastalari": [int(h) for h in dis["dis_dogrulama"][1]],
                    "dis_test_hastalari": [int(h) for h in dis["dis_test"][1]],
                    "egitim_karo": int(len(egitim)), "ic_dogrulama_karo": int(len(ic_dog)),
                    "dis_test_karo": int(len(dis["dis_test"][0]))},
        "egitim": {"mimari": "resnet18", "on_egitim": "IMAGENET1K_V1",
                   "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-4,
                   "yigin": YIGIN, "tur": TUR, "secilen_tur": secilen_tur,
                   "model_secim_olcutu": "ic_dogrulama_auc",
                   "artirma": "yatay+dikey cevirme", "renk_artirma": False,
                   "tur_gecmisi": tur_gecmisi},
        "normalizasyon": {"kaynak": "yalniz egitim karolari", "ort": ort.tolist(),
                          "std": std.tolist()},
        "esik": {"deger": esik_ic, "secim": "ic dogrulamada ozgulluk 0,90",
                 "not": "sahada hedef merkezin etiketi yoktur; esik ic veriden secilir"},
        "olcumler": olcumler,
        "ortam": {"python": platform.python_version(), "torch": torch.__version__,
                  "sklearn": sklearn.__version__, "numpy": np.__version__,
                  "aygit": str(aygit), "platform": platform.platform()},
        "sure_sn": round(time.time() - t0, 1),
    }
    with open(f"modeller/{ad_kok}.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"  -> modeller/{ad_kok}.pt · modeller/{ad_kok}.json · "
          f"sonuclar/skorlar_{ad_kok}.csv", flush=True)
    print(f"  -> ic {olcumler['ic_dogrulama']['auc']:.4f} | "
          f"dis test {olcumler['dis_test']['auc']:.4f} | {time.time()-t0:.0f} sn", flush=True)
    return manifest


isler = ([(k, t) for k in MERKEZLER for t in (42, 43, 44)] if args.hepsi
         else [(args.kat, args.tohum)])
for kat, tohum in isler:
    print(f"\n{'='*74}\nKAT {kat} | TOHUM {tohum}\n{'='*74}", flush=True)
    egit_kaydet(kat, tohum)
