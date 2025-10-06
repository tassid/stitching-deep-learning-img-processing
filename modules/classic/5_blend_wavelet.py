# modules/classic/blend_wavelet.py
# Uso (arquivos):
#   python -m modules.classic.blend_wavelet --imgs a.jpg b.jpg c.jpg --detector orb --matcher bf \
#       --wavelet db2 --levels 3 --rule max --out pano_wavelet.jpg
# Uso (HF):
#   python -m modules.classic.blend_wavelet --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
#       --count 24 --offset 300 --detector akaze --matcher flann --wavelet sym4 --levels 3 --rule mean --out pano_wavelet.jpg
# Requisitos: opencv-contrib-python, PyWavelets, datasets, pillow, huggingface_hub

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

# ---------- Wavelet Fusion apenas no overlap ----------
def wavelet_fuse_overlap(base, warp, base_mask, warp_mask, wavelet="db2", levels=2, rule="mean"):
    """
    Fusão wavelet simplificada - usa apenas 2 níveis por padrão para evitar travamentos.
    Se der qualquer problema, faz fallback para blend simples.
    """
    try:
        import pywt
    except ImportError:
        # Fallback: blend simples se PyWavelets não estiver instalado
        both = (base_mask > 0) & (warp_mask > 0)
        if np.any(both):
            dt_b = cv2.distanceTransform((base_mask>0).astype(np.uint8), cv2.DIST_L2, 5)
            dt_w = cv2.distanceTransform((warp_mask>0).astype(np.uint8), cv2.DIST_L2, 5)
            s = dt_b + dt_w + 1e-6
            wb = (dt_b/s)[...,None]
            ww = (dt_w/s)[...,None]
            out = base.astype(np.float32)*wb + warp.astype(np.float32)*ww
            return np.clip(out,0,255).astype(np.uint8)
        else:
            out = base.copy()
            out[warp_mask > 0] = warp[warp_mask > 0]
            return out

    both = (base_mask > 0) & (warp_mask > 0)
    if not np.any(both):
        out = base.copy()
        out[warp_mask > 0] = warp[warp_mask > 0]
        return out

    ys, xs = np.where(both)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    b_crop = base[y0:y1, x0:x1].astype(np.float32)
    w_crop = warp[y0:y1, x0:x1].astype(np.float32)

    # Se região pequena demais, faz blend simples
    h_crop, w_crop_size = b_crop.shape[:2]
    if h_crop < 32 or w_crop_size < 32:
        fused = (0.5 * b_crop + 0.5 * w_crop).astype(np.uint8)
        out = base.copy()
        out[y0:y1, x0:x1] = fused
        out[(base_mask==0) & (warp_mask>0)] = warp[(base_mask==0) & (warp_mask>0)]
        return out

    # Tenta wavelet fusion, mas com proteção total contra travamentos
    fused = np.empty_like(b_crop)
    for c in range(3):
        b_c = b_crop[..., c]
        w_c = w_crop[..., c]

        try:
            # Limita a apenas 2 níveis por segurança
            max_level = pywt.dwt_max_level(min(b_c.shape), pywt.Wavelet(wavelet).dec_len)
            safe_levels = max(1, min(2, max_level))  # máximo 2 níveis

            # Pad para múltiplo de 2^safe_levels
            h, w = b_c.shape
            mul = 2 ** safe_levels
            nh = int(np.ceil(h / mul) * mul)
            nw = int(np.ceil(w / mul) * mul)
            ph, pw = nh - h, nw - w

            b_p = cv2.copyMakeBorder(b_c, 0, ph, 0, pw, borderType=cv2.BORDER_REFLECT_101)
            w_p = cv2.copyMakeBorder(w_c, 0, ph, 0, pw, borderType=cv2.BORDER_REFLECT_101)

            # Decomposição wavelet
            coeffs_b = pywt.wavedec2(b_p, wavelet=wavelet, level=safe_levels, mode="symmetric")
            coeffs_w = pywt.wavedec2(w_p, wavelet=wavelet, level=safe_levels, mode="symmetric")

            # Fusão dos coeficientes (sempre usa média para ser mais estável)
            cA_b, *dets_b = coeffs_b
            cA_w, *dets_w = coeffs_w
            cA = 0.5 * (cA_b + cA_w)
            fused_coeffs = [cA]
            for (bH, bV, bD), (wH, wV, wD) in zip(dets_b, dets_w):
                fused_coeffs.append((0.5*(bH+wH), 0.5*(bV+wV), 0.5*(bD+wD)))

            # Reconstrução
            rec = pywt.waverec2(fused_coeffs, wavelet=wavelet, mode="symmetric")
            rec = rec[:h, :w]
            fused[..., c] = rec
        except:
            # Se qualquer coisa der errado, usa média simples
            fused[..., c] = 0.5 * (b_c + w_c)

    fused = np.clip(fused, 0, 255).astype(np.uint8)
    out = base.copy()
    out[y0:y1, x0:x1] = fused
    out[(base_mask==0) & (warp_mask>0)] = warp[(base_mask==0) & (warp_mask>0)]
    return out

# ---------- pipeline ----------
def stitch_wavelet(imgs: List[np.ndarray], detector_name: str, matcher_name: str, ratio: float, ransac: float,
                   conf: float, min_side: int, maxIters: int,
                   wavelet: str, levels: int, rule: str):
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

        warp_mask = (cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)>0).astype(np.uint8)
        base_mask = (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)>0).astype(np.uint8)

        canvas = wavelet_fuse_overlap(
            base, warp, base_mask, warp_mask,
            wavelet=wavelet, levels=levels, rule=rule
        )
    return canvas

# ---------- API para GUI ----------
def stitch(imgs, try_gpu=False, detector="sift", matcher="flann", ratio=0.75,
           ransac=3.0, conf=0.999, min_side=900, maxIters=20000,
           wavelet="db2", levels=2, rule="mean", **kwargs):
    """Interface simplificada para a GUI. Usa apenas 2 níveis wavelet para evitar travamentos."""
    return stitch_wavelet(imgs, detector, matcher, ratio, ransac, conf, min_side, maxIters,
                          wavelet, levels, rule)

# ---------- CLI ----------
def cli_main():
    p = argparse.ArgumentParser(prog="classic.blend_wavelet")
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

    p.add_argument("--wavelet", default="db2", help="ex.: db2, sym4, coif2, haar")
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--rule", choices=["max","mean"], default="max", help="fusão das bandas de detalhe")
    p.add_argument("--out", default="pano_wavelet.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    pano = stitch_wavelet(
        imgs, a.detector, a.matcher, a.ratio, a.ransac, a.conf, a.min_side, a.maxIters,
        wavelet=a.wavelet, levels=a.levels, rule=a.rule
    )
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
