# modules/classic/11_blend_multifocal.py
# Fusão multi-focal clássica (focus stacking) após alinhamento por features.
# Uso (arquivos):
#   python -m modules.classic.11_blend_multifocal --imgs a.jpg b.jpg c.jpg --detector sift --matcher flann \
#       --focus lapvar --levels 5 --out pano_multifocal.jpg
# Uso (HF):
#   python -m modules.classic.11_blend_multifocal --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
#       --count 24 --offset 300 --detector orb --matcher bf --focus tenengrad --levels 5 --out pano_multifocal.jpg
# Requisitos: opencv-contrib-python, datasets, pillow, huggingface_hub

import argparse, sys
from typing import List, Tuple
import numpy as np
import cv2

# ---------- utils de IO ----------
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

# ---------- detecção/descrição + homografia ----------
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

# ---------- medidores de foco ----------
def focus_map_lapvar(img_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    lap = cv2.Laplacian(img_gray, cv2.CV_32F, ksize=ksize)
    fm = lap * lap
    return cv2.GaussianBlur(fm, (ksize|1, ksize|1), 0)

def focus_map_tenengrad(img_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=ksize)
    fm = gx*gx + gy*gy
    return cv2.GaussianBlur(fm, (ksize|1, ksize|1), 0)

def normalize01(m: np.ndarray) -> np.ndarray:
    mn, mx = float(m.min()), float(m.max())
    if mx - mn < 1e-12:
        return np.zeros_like(m, dtype=np.float32)
    return ((m - mn) / (mx - mn)).astype(np.float32)

# ---------- pirâmides ----------
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
        lp.append(cv2.subtract(gp[i], up))
    lp.append(gp[-1])
    return lp

def _pyr_reconstruct(lp: List[np.ndarray]):
    img = lp[-1]
    for lev in range(len(lp)-2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(lp[lev].shape[1], lp[lev].shape[0]))
        img = cv2.add(img, lp[lev])
    return img

def _make_weight_pyr(mask_float: np.ndarray, levels: int):
    gp = [mask_float]
    cur = mask_float
    for _ in range(levels):
        cur = cv2.pyrDown(cur)
        gp.append(cur)
    for i in range(len(gp)):
        gp[i] = np.clip(cv2.GaussianBlur(gp[i], (0,0), 0.5), 0.0, 1.0)
    return gp

# ---------- fusão multifocal na região de overlap ----------
def multifocus_fuse_overlap(base, warp, base_mask, warp_mask, levels=5, method="lapvar", ksize=3):
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

    gb = cv2.cvtColor(b.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gw = cv2.cvtColor(w.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if method.lower() == "tenengrad":
        fb = focus_map_tenengrad(gb, ksize=ksize)
        fw = focus_map_tenengrad(gw, ksize=ksize)
    else:
        fb = focus_map_lapvar(gb, ksize=ksize)
        fw = focus_map_lapvar(gw, ksize=ksize)

    fb = normalize01(fb); fw = normalize01(fw)
    # pesos suaves e normalizados (evita bordas duras)
    sumw = fb + fw + 1e-6
    wb = fb / sumw
    ww = fw / sumw

    # pirâmide laplaciana com pesos multi-escala
    fused = np.empty_like(b)
    for c in range(3):
        Lb = _lap_pyr(b[...,c], levels)
        Lw = _lap_pyr(w[...,c], levels)
        Wb = _make_weight_pyr(wb, levels)
        Ww = _make_weight_pyr(ww, levels)
        Lf = [Lb[i]*Wb[i] + Lw[i]*Ww[i] for i in range(levels)]
        top = Lb[-1]*Wb[-1] + Lw[-1]*Ww[-1]
        Lf.append(top)
        rec = _pyr_reconstruct(Lf)
        fused[...,c] = rec

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    out = base.copy()
    out[y0:y1, x0:x1] = fused
    out[(base_mask==0) & (warp_mask>0)] = warp[(base_mask==0) & (warp_mask>0)]
    return out

# ---------- pipeline ----------
def stitch_multifocal(imgs: List[np.ndarray], detector_name: str, matcher_name: str, ratio: float, ransac: float,
                      conf: float, min_side: int, maxIters: int, levels: int, focus: str, ksize: int):
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

        canvas = multifocus_fuse_overlap(base, warp, base_mask, warp_mask,
                                         levels=levels, method=focus, ksize=ksize)
    return canvas

# ---------- API para GUI ----------
def stitch(imgs, try_gpu=False, detector="sift", matcher="flann", ratio=0.75,
           ransac=3.0, conf=0.999, min_side=900, maxIters=20000,
           levels=5, focus="lapvar", ksize=3, **kwargs):
    """Interface simplificada para a GUI."""
    return stitch_multifocal(imgs, detector, matcher, ratio, ransac, conf, min_side, maxIters,
                             levels, focus, ksize)

# ---------- CLI ----------
def cli_main():
    p = argparse.ArgumentParser(prog="classic.11_blend_multifocal")
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

    p.add_argument("--focus", choices=["lapvar","tenengrad"], default="lapvar",
                   help="medida de foco: variância do Laplaciano (lapvar) ou Tenengrad (sobel)")
    p.add_argument("--ksize", type=int, default=3, help="ksize dos filtros (Laplaciano/Sobel)")
    p.add_argument("--levels", type=int, default=5, help="níveis da pirâmide Laplaciana")
    p.add_argument("--out", default="pano_multifocal.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    pano = stitch_multifocal(
        imgs, a.detector, a.matcher, a.ratio, a.ransac, a.conf,
        a.min_side, a.maxIters, a.levels, a.focus, a.ksize
    )
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
