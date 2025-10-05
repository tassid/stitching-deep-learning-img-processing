# 2_manual_sift_ransac_feather.py
# uso (arquivos locais):
#   python main.py classic.2_manual_sift_ransac_feather --imgs a.jpg b.jpg c.jpg --ratio 0.75 --out pano_manual.jpg
# uso (dataset HF):
#  python -m modules.classic.2_manual_sift_ransac_feather --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train --count 24 --offset 300 --ratio 0.75 --out pano_manual.jpg

import cv2, numpy as np, argparse, sys
from typing import List

# ---- utilidades ----

def _require_sift():
    if not hasattr(cv2, "SIFT_create"):
        sys.exit("Esta técnica requer SIFT: instale opencv-contrib-python.")

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


def resize_min_side(img, min_side: int):
    if min_side <= 0:
        return img
    h, w = img.shape[:2]
    s = min(h, w)
    if s <= min_side:
        return img
    scale = min_side / s
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)


# ---- núcleo: matching + homografia ----

def match_and_h(a, b, ratio=0.75, ransac_thresh=3.0, conf=0.999, maxIters=10000):
    _require_sift()
    sift = cv2.SIFT_create()
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return None, []
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = matcher.knnMatch(da, db, k=2)
    good = []
    for m, n in knn:
        if m.distance < n.distance * ratio:
            good.append(m)
    if len(good) < 4:
        return None, []
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_thresh, maxIters=maxIters, confidence=conf)
    return H, inl


# ---- blending por feather (2 lados) ----

def feather_blend(base, warp, base_mask, warp_mask):
    # distância até borda para cada máscara
    base_dt = cv2.distanceTransform((base_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    warp_dt = cv2.distanceTransform((warp_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    wsum = base_dt + warp_dt + 1e-6
    wb = (base_dt / wsum)[..., None]
    ww = (warp_dt / wsum)[..., None]
    out = base.astype(np.float32) * wb + warp.astype(np.float32) * ww
    return np.clip(out, 0, 255).astype(np.uint8)


# ---- mosaico incremental ----

def stitch_sequence(imgs: List[np.ndarray], ratio: float, min_side: int):
    canvas = resize_min_side(imgs[0], min_side)
    for i in range(1, len(imgs)):
        cur = resize_min_side(imgs[i], min_side)
        H, inl = match_and_h(cur, canvas, ratio=ratio)
        if H is None:
            raise SystemExit(f"homografia insuficiente na imagem {i} (ratio={ratio})")
        h, w = cur.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        wc = cv2.perspectiveTransform(corners, H)
        cc = np.vstack((wc, np.float32([[0, 0], [canvas.shape[1], 0], [canvas.shape[1], canvas.shape[0]], [0, canvas.shape[0]]]).reshape(-1, 1, 2)))
        [xmin, ymin] = np.int32(cc.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(cc.max(axis=0).ravel() + 0.5)
        t = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float32)

        warp = cv2.warpPerspective(cur, t @ H, (xmax - xmin, ymax - ymin))
        base = np.zeros_like(warp)
        base[-ymin: -ymin + canvas.shape[0], -xmin: -xmin + canvas.shape[1]] = canvas

        warp_mask = (cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        base_mask = (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
        # suaviza/expande levemente para evitar costuras duras
        k = np.ones((3, 3), np.uint8)
        warp_mask = cv2.morphologyEx(warp_mask, cv2.MORPH_DILATE, k)
        base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_DILATE, k)

        canvas = feather_blend(base, warp, base_mask, warp_mask)
    return canvas


# ---- API para GUI ----

def stitch(imgs, try_gpu=False, ratio=0.75, min_side=900, **kwargs):
    """Interface simplificada para a GUI."""
    return stitch_sequence(imgs, ratio=ratio, min_side=min_side)

# ---- CLI ----

def cli_main():
    p = argparse.ArgumentParser(prog="classic.2_manual_sift_ransac_feather")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF, ex: miroslavjaros/Video-To-Dataset-Orchard-Apple")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio-test (0.6–0.85)")
    p.add_argument("--min_side", type=int, default=900, help="resize do lado menor (0 = sem)")
    p.add_argument("--out", default="pano_manual.jpg")
    a = p.parse_args()

    imgs = [cv2.imread(x) for x in a.imgs] if a.imgs else load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    pano = stitch_sequence(imgs, ratio=a.ratio, min_side=a.min_side)
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
