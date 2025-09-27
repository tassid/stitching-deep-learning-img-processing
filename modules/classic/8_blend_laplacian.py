# modules/classic/blend_laplacian.py
# Uso (arquivos):
#   python -m modules.classic.blend_laplacian --imgs a.jpg b.jpg c.jpg --detector orb --matcher bf \
#       --levels 5 --min_side 900 --out pano_laplacian.jpg
# Uso (HF):
#   python -m modules.classic.blend_laplacian --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
#       --count 24 --offset 300 --detector akaze --matcher flann --levels 5 --out pano_laplacian.jpg
# Requisitos: opencv-contrib-python, datasets, pillow, huggingface_hub

import argparse, sys
from typing import List, Tuple
import numpy as np
import cv2

# ---------- utils ----------
def pil_to_bgr(pil_img):
    import numpy as np
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def load_from_hf(name: str, split: str, count: int, offset: int, shuffle: bool) -> List[np.ndarray]:
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
    if min_side <= 0: return img
    h, w = img.shape[:2]
    s = min(h, w)
    if s <= min_side: return img
    scale = min_side / s
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# ---------- features + H ----------
def build_detector(name: str):
    name = name.lower()
    if name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            sys.exit("SIFT requer opencv-contrib-python.")
        return cv2.SIFT_create()
    if name == "orb":
        return cv2.ORB_create(nfeatures=4000)
    if name == "akaze":
        return cv2.AKAZE_create()
    sys.exit("detector inválido: use sift|orb|akaze")

def build_matcher(matcher: str, detector: str):
    matcher = matcher.lower(); detector = detector.lower()
    if matcher == "bf":
        if detector == "sift":
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    if matcher == "flann":
        if detector == "sift":
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=64)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:  # ORB/AKAZE → LSH
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=64)
            return cv2.FlannBasedMatcher(index_params, search_params)
    sys.exit("matcher inválido: use bf|flann")

def find_h(a, b, detector, matcher, ratio, ransac_thresh, conf, maxIters) -> Tuple[np.ndarray, np.ndarray, list]:
    ka, da = detector.detectAndCompute(a, None)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return None, None, []
    knn = matcher.knnMatch(da, db, k=2)
    good = [m for m,n in knn if m.distance < n.distance * ratio]
    if len(good) < 4:
        return None, None, []
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, ransac_thresh, maxIters=maxIters, confidence=conf)
    return H, inl, good

# ---------- Laplacian Pyramid Blending (apenas no overlap) ----------
def _gauss_pyr(img, levels: int):
    pyr = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        pyr.append(img)
    return pyr

def _lap_pyr(img, levels: int):
    gp = _gauss_pyr(img, levels)
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lap = cv2.subtract(gp[i], up)
        lp.append(lap)
    lp.append(gp[-1])  # topo gaussiano
    return lp

def _pyr_reconstruct(lp: List[np.ndarray]):
    img = lp[-1]
    for lev in range(len(lp)-2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(lp[lev].shape[1], lp[lev].shape[0]))
        img = cv2.add(img, lp[lev])
    return img

def _make_weight_pyr(mask_float: np.ndarray, levels: int):
    # máscara suave 0..1 → pirâmide gaussiana de pesos
    gp = [mask_float]
    cur = mask_float
    for _ in range(levels):
        cur = cv2.pyrDown(cur)
        gp.append(cur)
    # normaliza cada nível para [0,1]
    for i in range(len(gp)):
        m = gp[i]
        m = cv2.GaussianBlur(m, (0,0), 0.5)
        gp[i] = np.clip(m, 0.0, 1.0)
    return gp

def laplacian_blend_overlap(base, warp, base_mask, warp_mask, levels=5):
    # região de overlap
    both = (base_mask>0) & (warp_mask>0)
    if not np.any(both):
        out = base.copy()
        out[warp_mask>0] = warp[warp_mask>0]
        return out

    ys, xs = np.where(both)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1

    b = base[y0:y1, x0:x1].astype(np.float32)
    w = warp[y0:y1, x0:x1].astype(np.float32)

    # pesos a partir das máscaras (suavizados)
    bm = (base_mask[y0:y1, x0:x1].astype(np.float32) / 255.0)
    wm = (warp_mask[y0:y1, x0:x1].astype(np.float32) / 255.0)

    # normalizar para soma=1 no overlap, evitando div por zero
    sumw = bm + wm
    sumw[sumw == 0] = 1.0
    wb = bm / sumw
    ww = wm / sumw

    # pirâmides de imagem por canal
    fused = np.empty_like(b)
    for c in range(3):
        b_c = b[..., c]
        w_c = w[..., c]
        # laplacianas
        Lb = _lap_pyr(b_c, levels)
        Lw = _lap_pyr(w_c, levels)
        # pesos (gaussianos) por nível
        Wb = _make_weight_pyr(wb, levels)
        Ww = _make_weight_pyr(ww, levels)
        # blend em cada nível
        Lf = []
        for i in range(levels):
            l = Lb[i]*Wb[i] + Lw[i]*Ww[i]
            Lf.append(l)
        # topo gaussiano
        top = Lb[-1]*Wb[-1] + Lw[-1]*Ww[-1]
        Lf.append(top)
        rec = _pyr_reconstruct(Lf)
        fused[..., c] = rec

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    out = base.copy()
    out[y0:y1, x0:x1] = fused
    # onde só warp existe, copia warp
    out[(base_mask==0) & (warp_mask>0)] = warp[(base_mask==0) & (warp_mask>0)]
    return out

# ---------- pipeline ----------
def stitch_laplacian(imgs: List[np.ndarray], detector_name: str, matcher_name: str, ratio: float, ransac: float,
                     conf: float, min_side: int, maxIters: int, levels: int):
    det = build_detector(detector_name)
    mat = build_matcher(matcher_name, detector_name)

    canvas = resize_min_side(imgs[0], min_side)
    for i in range(1, len(imgs)):
        cur = resize_min_side(imgs[i], min_side)
        H, inl, good = find_h(cur, canvas, det, mat, ratio, ransac, conf, maxIters)
        if H is None:
            raise SystemExit(f"homografia insuficiente na imagem {i} (matches={len(good)})")

        h, w = cur.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        wc = cv2.perspectiveTransform(corners, H)
        cc = np.vstack((wc, np.float32([[0,0],[canvas.shape[1],0],[canvas.shape[1],canvas.shape[0]],[0,canvas.shape[0]]]).reshape(-1,1,2)))
        [xmin,ymin] = np.int32(cc.min(axis=0).ravel()-0.5)
        [xmax,ymax] = np.int32(cc.max(axis=0).ravel()+0.5)
        T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)

        warp = cv2.warpPerspective(cur, T@H, (xmax-xmin, ymax-ymin))
        base = np.zeros_like(warp)
        base[-ymin:-ymin+canvas.shape[0], -xmin:-xmin+canvas.shape[1]] = canvas

        warp_mask = (cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)>0).astype(np.uint8)*255
        base_mask = (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)>0).astype(np.uint8)*255

        canvas = laplacian_blend_overlap(base, warp, base_mask, warp_mask, levels=levels)
    return canvas

# ---------- CLI ----------
def cli_main():
    p = argparse.ArgumentParser(prog="classic.blend_laplacian")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--detector", choices=["sift","orb","akaze"], default="sift")
    p.add_argument("--matcher", choices=["bf","flann"], default="flann")
    p.add_argument("--ratio", type=float, default=0.75)
    p.add_argument("--ransac", type=float, default=3.0)
    p.add_argument("--conf", type=float, default=0.999)
    p.add_argument("--min_side", type=int, default=900)
    p.add_argument("--maxIters", type=int, default=20000)

    p.add_argument("--levels", type=int, default=5, help="nº de níveis da pirâmide Laplaciana")
    p.add_argument("--out", default="pano_laplacian.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    pano = stitch_laplacian(
        imgs, a.detector, a.matcher, a.ratio, a.ransac, a.conf, a.min_side, a.maxIters,
        levels=a.levels
    )
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
