# modules/classic/ecc_alignment.py
# Alinhamento incremental com ECC (Enhanced Correlation Coefficient)
# Bom para cenas planares / variação fotométrica moderada.
#
# Uso (arquivos locais):
#   python -m modules.classic.ecc_alignment --imgs a.jpg b.jpg c.jpg --warp affine --iters 300 --eps 1e-6 --out pano_ecc.jpg
# Uso (HuggingFace):
#   python -m modules.classic.ecc_alignment --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
#       --count 24 --offset 300 --warp homography --out pano_ecc.jpg
#
# Requisitos: opencv-contrib-python, datasets, pillow, huggingface_hub

import argparse, sys
from typing import List
import numpy as np
import cv2

# ---------- util ----------
def pil_to_bgr(pil_img):
    import numpy as np
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def load_from_hf(name: str, split: str, count: int, offset: int, shuffle: bool) -> List[np.ndarray]:
    from datasets import load_dataset
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

def resize_min_side(img, min_side: int):
    if min_side <= 0: return img
    h, w = img.shape[:2]
    s = min(h, w)
    if s <= min_side: return img
    scale = min_side / s
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# ---------- ECC core ----------
def ecc_register(mov_gray, fix_gray, warp_mode: str, iters: int, eps: float):
    wm_name = warp_mode.lower()
    if wm_name == "translation":
        wm = cv2.MOTION_TRANSLATION; W = np.eye(2, 3, dtype=np.float32)
    elif wm_name == "euclidean":
        wm = cv2.MOTION_EUCLIDEAN;  W = np.eye(2, 3, dtype=np.float32)
    elif wm_name == "affine":
        wm = cv2.MOTION_AFFINE;     W = np.eye(2, 3, dtype=np.float32)
    elif wm_name == "homography":
        wm = cv2.MOTION_HOMOGRAPHY; W = np.eye(3, 3, dtype=np.float32)
    else:
        sys.exit("warp inválido: translation|euclidean|affine|homography")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    cc, W = cv2.findTransformECC(fix_gray, mov_gray, W, wm, criteria)
    return wm, W

def warp_by_mode(img, W, wm, size):
    if wm == cv2.MOTION_HOMOGRAPHY:
        return cv2.warpPerspective(img, W, size, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        return cv2.warpAffine(img, W, size, flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

# ---------- blending feather ----------
def feather_blend(base, warp, mask_base, mask_warp):
    k = np.ones((3,3), np.uint8)
    mask_base = cv2.morphologyEx(mask_base, cv2.MORPH_DILATE, k)
    mask_warp = cv2.morphologyEx(mask_warp, cv2.MORPH_DILATE, k)
    dt_b = cv2.distanceTransform((mask_base>0).astype(np.uint8), cv2.DIST_L2, 5)
    dt_w = cv2.distanceTransform((mask_warp>0).astype(np.uint8), cv2.DIST_L2, 5)
    wsum = dt_b + dt_w + 1e-6
    wb = (dt_b/wsum)[...,None]; ww = (dt_w/wsum)[...,None]
    out = base.astype(np.float32)*wb + warp.astype(np.float32)*ww
    return np.clip(out,0,255).astype(np.uint8)

# ---------- pipeline ----------
def stitch_ecc(imgs: List[np.ndarray], warp_mode: str, min_side: int, iters: int, eps: float):
    canvas = resize_min_side(imgs[0], min_side)
    gray_c = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    for i in range(1, len(imgs)):
        cur = resize_min_side(imgs[i], min_side)
        gray_m = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

        wm, W = ecc_register(gray_m, gray_c, warp_mode, iters, eps)

        h, w = cur.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        if wm == cv2.MOTION_HOMOGRAPHY:
            wc = cv2.perspectiveTransform(corners, W)
        else:
            W3 = np.vstack([W, [0,0,1]]).astype(np.float32)
            wc = cv2.perspectiveTransform(corners, W3)

        cc = np.vstack((wc, np.float32([[0,0],[canvas.shape[1],0],[canvas.shape[1],canvas.shape[0]],[0,canvas.shape[0]]]).reshape(-1,1,2)))
        [xmin,ymin] = np.int32(cc.min(axis=0).ravel()-0.5)
        [xmax,ymax] = np.int32(cc.max(axis=0).ravel()+0.5)
        T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)

        # compor transformação com deslocamento T
        if wm == cv2.MOTION_HOMOGRAPHY:
            Wt = T @ W
            warp = cv2.warpPerspective(cur, Wt, (xmax-xmin, ymax-ymin))
        else:
            T2 = T[:2,:]
            Wt = T2 @ np.vstack([W,[0,0,1]])
            warp = cv2.warpAffine(cur, Wt, (xmax-xmin, ymax-ymin))

        base = np.zeros_like(warp)
        base[-ymin:-ymin+canvas.shape[0], -xmin:-xmin+canvas.shape[1]] = canvas

        mask_w = (cv2.cvtColor(warp,  cv2.COLOR_BGR2GRAY)>0).astype(np.uint8)*255
        mask_b = (cv2.cvtColor(base,  cv2.COLOR_BGR2GRAY)>0).astype(np.uint8)*255

        canvas = feather_blend(base, warp, mask_b, mask_w)
        gray_c = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    return canvas

# ---------- API para GUI ----------
def stitch(imgs, try_gpu=False, warp_mode="affine", min_side=900, iters=300, eps=1e-6, **kwargs):
    """Interface simplificada para a GUI."""
    return stitch_ecc(imgs, warp_mode=warp_mode, min_side=min_side, iters=iters, eps=eps)

# ---------- CLI ----------
def cli_main():
    p = argparse.ArgumentParser(prog="classic.ecc_alignment")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--warp", choices=["translation","euclidean","affine","homography"], default="affine")
    p.add_argument("--min_side", type=int, default=900)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--out", default="pano_ecc.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    pano = stitch_ecc(imgs, warp_mode=a.warp, min_side=a.min_side, iters=a.iters, eps=a.eps)
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
