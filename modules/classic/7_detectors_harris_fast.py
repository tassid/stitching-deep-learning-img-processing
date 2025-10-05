# modules/classic/detectors_harris_fast.py
# Demonstração de detectores clássicos: Harris e FAST
# Uso (arquivos locais):
#   python -m modules.classic.detectors_harris_fast --imgs a.jpg b.jpg c.jpg --detector harris --threshold 0.04 --max_kp 1000
# Uso (dataset HF):
#   python -m modules.classic.detectors_harris_fast --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
#       --count 12 --offset 300 --detector fast --threshold 15 --nonmax --max_kp 1500
# Saídas: imagens anotadas em --out_dir (padrão: out_detectors)
# Requisitos: opencv-contrib-python (ou opencv-python), datasets, pillow, huggingface_hub

import argparse, sys, os
from typing import List, Tuple
import numpy as np
import cv2

# ---------------- utilidades comuns ----------------

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
    if end - offset < 1:
        sys.exit(f"Nenhuma imagem suficiente; split '{split}' tem {len(ds)} registros.")
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
    if len(imgs) < 1:
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

# ---------------- Harris ----------------

def detect_harris(img_bgr: np.ndarray, thresh_alpha: float = 0.04, ksize: int = 3,
                  block_size: int = 2, max_kp: int = 1000, nonmax_radius: int = 8) -> List[cv2.KeyPoint]:
    """
    Harris clássico: resposta = cornerHarris(gray, blockSize, ksize, k=alpha)
    - thresh_alpha: fator do Harris (k), não limiar. O limiar é relativo ao máximo da resposta.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)
    R = cv2.cornerHarris(gray_f, blockSize=block_size, ksize=ksize, k=thresh_alpha)
    # normaliza
    Rn = cv2.normalize(R, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # nonmax suppression aproximada: pega picos locais via dilatação e máscara
    dil = cv2.dilate(Rn, np.ones((3,3), np.uint8))
    peaks = (Rn == dil) & (Rn > 0.01)  # 0.01 é um limiar inicial baixo
    ys, xs = np.where(peaks)
    scores = Rn[ys, xs]
    # ordena por resposta
    idx = np.argsort(-scores)[:max_kp*5]  # pega mais e filtra por vizinhança
    pts = np.stack([xs[idx], ys[idx], scores[idx]], axis=1)

    # non-maximum suppression por distância (greedy)
    selected = []
    used = np.zeros(len(pts), dtype=bool)
    for i in range(len(pts)):
        if used[i]:
            continue
        selected.append(pts[i])
        # suprime vizinhos próximos
        dx = pts[:,0] - pts[i,0]
        dy = pts[:,1] - pts[i,1]
        used = used | ((dx*dx + dy*dy) <= nonmax_radius*nonmax_radius)
        if len(selected) >= max_kp:
            break

    kps = [cv2.KeyPoint(float(x), float(y), 7.0, response=float(s)) for x, y, s in selected]
    return kps

# ---------------- FAST ----------------

def detect_fast(img_bgr: np.ndarray, threshold: int = 10, nonmax: bool = True, max_kp: int = 1000) -> List[cv2.KeyPoint]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create(threshold=int(threshold), nonmaxSuppression=bool(nonmax))
    kps = fast.detect(gray, None) or []
    # ordena por resposta, corta
    kps = sorted(kps, key=lambda k: k.response if k.response is not None else 0, reverse=True)[:max_kp]
    return kps

# ---------------- desenho ----------------

def draw_keypoints(img_bgr: np.ndarray, kps: List[cv2.KeyPoint], color=(0,255,0)) -> np.ndarray:
    out = img_bgr.copy()
    return cv2.drawKeypoints(out, kps, None, color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ---------------- Stitching com Harris/FAST ----------------

def stitch_with_detector(imgs: List[np.ndarray], detector: str = "harris",
                         ratio: float = 0.75, ransac_thresh: float = 3.0,
                         min_side: int = 900, max_kp: int = 1000,
                         alpha: float = 0.04, threshold: int = 10,
                         nonmax: bool = True) -> np.ndarray:
    """
    Stitching usando Harris ou FAST como detector de features.
    Usa SIFT para descritores e matching.
    """
    imgs_r = [resize_min_side(im, min_side) for im in imgs]

    # Inicializar SIFT para descritores
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def match_pair(im_a, im_b):
        # Detectar keypoints com Harris ou FAST
        if detector == "harris":
            kps_a = detect_harris(im_a, thresh_alpha=alpha, max_kp=max_kp)
            kps_b = detect_harris(im_b, thresh_alpha=alpha, max_kp=max_kp)
        else:  # FAST
            kps_a = detect_fast(im_a, threshold=threshold, nonmax=nonmax, max_kp=max_kp)
            kps_b = detect_fast(im_b, threshold=threshold, nonmax=nonmax, max_kp=max_kp)

        if len(kps_a) < 4 or len(kps_b) < 4:
            raise RuntimeError(f"Keypoints insuficientes: A={len(kps_a)}, B={len(kps_b)}")

        # Computar descritores SIFT nas posições detectadas
        gray_a = cv2.cvtColor(im_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(im_b, cv2.COLOR_BGR2GRAY)
        kps_a, desc_a = sift.compute(gray_a, kps_a)
        kps_b, desc_b = sift.compute(gray_b, kps_b)

        if desc_a is None or desc_b is None or len(desc_a) < 4 or len(desc_b) < 4:
            raise RuntimeError("Descritores insuficientes após SIFT.compute()")

        # Matching com ratio test
        knn = bf.knnMatch(desc_b, desc_a, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < 8:
            raise RuntimeError(f"Correspondências insuficientes: {len(good)}")

        # Estimar homografia
        pts_b = np.float32([kps_b[m.queryIdx].pt for m in good])
        pts_a = np.float32([kps_a[m.trainIdx].pt for m in good])
        H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, ransac_thresh)

        if H is None:
            raise RuntimeError("Falha ao estimar homografia (RANSAC)")

        return H

    def warp_merge(base, img, H):
        h1, w1 = base.shape[:2]
        h2, w2 = img.shape[:2]
        corners_img = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        corners_img_w = cv2.perspectiveTransform(corners_img, H)
        corners_base = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        all_pts = np.vstack((corners_img_w, corners_base))
        xmin, ymin = np.floor(all_pts.min(axis=0).ravel()).astype(int)
        xmax, ymax = np.ceil(all_pts.max(axis=0).ravel()).astype(int)
        tx, ty = -xmin, -ymin
        T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=float)
        size = (xmax - xmin, ymax - ymin)
        base_w = cv2.warpPerspective(base, T, size)
        img_w = cv2.warpPerspective(img, T @ H, size)
        mask_b = (base_w.sum(axis=2) > 0).astype("uint8") * 255
        mask_i = (img_w.sum(axis=2) > 0).astype("uint8") * 255
        overlap = cv2.bitwise_and(mask_b, mask_i)
        out = base_w.copy()
        out[mask_i > 0] = img_w[mask_i > 0]
        if overlap.any():
            mix = (base_w.astype(float) + img_w.astype(float)) / 2.0
            out[overlap > 0] = mix[overlap > 0].astype("uint8")
        return out

    # Stitching incremental
    pano = imgs_r[0]
    for i in range(1, len(imgs_r)):
        H = match_pair(pano, imgs_r[i])
        pano = warp_merge(pano, imgs_r[i], H)

    return pano

# ---------------- API para GUI ----------------

def stitch(imgs, try_gpu=False, detector="harris", ratio=0.75, ransac=3.0,
           min_side=900, max_kp=1000, alpha=0.04, threshold=10,
           nonmax=True, **kwargs):
    """
    Interface simplificada para a GUI.
    Usa Harris ou FAST como detector + SIFT descritores.
    """
    return stitch_with_detector(
        imgs, detector=detector, ratio=ratio, ransac_thresh=ransac,
        min_side=min_side, max_kp=max_kp, alpha=alpha,
        threshold=threshold, nonmax=nonmax
    )

# ---------------- CLI ----------------

def cli_main():
    p = argparse.ArgumentParser(prog="classic.detectors_harris_fast")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--detector", choices=["harris","fast"], default="harris")
    # HARRIS params
    p.add_argument("--alpha", type=float, default=0.04, help="parâmetro k do Harris (0.04 padrão)")
    p.add_argument("--block", type=int, default=2, help="Harris blockSize")
    p.add_argument("--ksize", type=int, default=3, help="Harris ksize (3,5...)")
    p.add_argument("--nms_radius", type=int, default=8, help="raio p/ supressão não-máxima (Harris)")
    # FAST params
    p.add_argument("--threshold", type=int, default=10, help="threshold do FAST")
    p.add_argument("--nonmax", action="store_true", help="habilita nonmax do FAST (desligado por padrão)")
    # comuns
    p.add_argument("--max_kp", type=int, default=1000, help="máximo de keypoints por imagem")
    p.add_argument("--min_side", type=int, default=900, help="redimensiona lado menor (0 = não)")
    p.add_argument("--out_dir", default="out_detectors")
    a = p.parse_args()

    # carregar imagens
    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    os.makedirs(a.out_dir, exist_ok=True)

    total = 0
    for i, im in enumerate(imgs):
        imr = resize_min_side(im, a.min_side)

        if a.detector == "harris":
            kps = detect_harris(
                imr, thresh_alpha=a.alpha, ksize=a.ksize, block_size=a.block,
                max_kp=a.max_kp, nonmax_radius=a.nms_radius
            )
        else:
            kps = detect_fast(
                imr, threshold=a.threshold, nonmax=a.nonmax, max_kp=a.max_kp
            )

        annotated = draw_keypoints(imr, kps)
        out_path = os.path.join(a.out_dir, f"{a.detector}_{i:03d}.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"[{i:03d}] {a.detector}: {len(kps)} pontos  -> {out_path}")
        total += len(kps)

    print(f"ok: {len(imgs)} imagens processadas, total de {total} keypoints.")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
