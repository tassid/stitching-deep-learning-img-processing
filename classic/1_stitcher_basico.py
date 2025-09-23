# 1_stitcher_basico.py
# uso (arquivos locais):
#   python 1_stitcher_basico.py --imgs a.jpg b.jpg c.jpg --modo panorama
# uso (dataset HuggingFace):
#   python 1_stitcher_basico.py --hf LivingOptics/hyperspectral-orchard --split train --count 5 --out pano.jpg

# falhou com status 1: não há sobreposição suficiente entre as imagens

import cv2, argparse, sys
from typing import List

def make_stitcher(modo: str, try_gpu: bool):
    mode = cv2.Stitcher_PANORAMA if modo == "panorama" else cv2.Stitcher_SCANS
    st = cv2.Stitcher_create(mode)
    st.setPanoConfidenceThresh(0.5)
    # tenta usar warper se existir
    try:
        Warper = getattr(cv2, "PyRotationWarper", None)
        if Warper and hasattr(st, "setWarper"):
            st.setWarper(Warper("spherical", 1.0))
    except Exception:
        pass
    return st

def pil_to_bgr(pil_img):
    import numpy as np
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def load_from_hf(name: str, split: str, count: int, offset: int, shuffle: bool) -> List:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Instale 'datasets' e 'Pillow': pip install datasets pillow")

    ds = load_dataset(name, split=split)
    if shuffle:
        ds = ds.shuffle(seed=42)

    end = min(len(ds), offset + count)
    if end - offset < 2:
        sys.exit(f"Preciso de >=2 imagens; split '{split}' tem {len(ds)} registros.")

    imgs = []
    for rec in ds.select(range(offset, end)):
        img = rec.get("image") or rec.get("rgb") or rec[list(rec.keys())[0]]
        try:
            imgs.append(pil_to_bgr(img))
        except Exception:
            if isinstance(img, str):
                im = cv2.imread(img)
                if im is not None:
                    imgs.append(im)

    if len(imgs) < 2:
        sys.exit("Não consegui ler imagens RGB desse dataset/split.")
    return imgs

p = argparse.ArgumentParser()
p.add_argument("--imgs", nargs="+", help="arquivos locais")
p.add_argument("--hf", help="ex: LivingOptics/hyperspectral-orchard")
p.add_argument("--split", default="train")
p.add_argument("--count", type=int, default=5)
p.add_argument("--offset", type=int, default=0)
p.add_argument("--shuffle", action="store_true")
p.add_argument("--modo", choices=["panorama", "scans"], default="panorama")
p.add_argument("--try_gpu", action="store_true")
p.add_argument("--out", default="pano_stitcher.jpg")
a = p.parse_args()

if not a.imgs and not a.hf:
    sys.exit("Use --imgs ... ou --hf <dataset>.")

imgs = [cv2.imread(x) for x in a.imgs] if a.imgs else load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
if any(im is None for im in imgs):
    sys.exit("Falha ao ler alguma imagem.")

st = make_stitcher(a.modo, a.try_gpu)
status, pano = st.stitch(imgs)

if status != cv2.STITCHER_OK:
    sys.exit(f"falhou com status {status}")

cv2.imwrite(a.out, pano)
print(f"ok: {a.out}")
