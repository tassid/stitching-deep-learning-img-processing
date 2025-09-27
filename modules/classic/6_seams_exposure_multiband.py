# modules/classic/seams_exposure_multiband.py
# Uso (arquivos):
#   python -m modules.classic.seams_exposure_multiband --imgs a.jpg b.jpg c.jpg --detector sift --matcher flann --out pano_seams.jpg
# Uso (HF):
#   python -m modules.classic.seams_exposure_multiband --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
#     --count 24 --offset 300 --detector orb --matcher bf --out pano_seams.jpg
# Requisitos: opencv-contrib-python==4.10.*, datasets, pillow, huggingface_hub

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

# ---------- helpers detail ----------
def make_seam_finder(kind: str = "dp"):
    kind = kind.lower()
    # GraphCut pode não estar disponível em algumas builds
    if kind == "graphcut":
        try:
            return cv2.detail_GraphCutSeamFinder("COST_COLOR")
        except Exception:
            pass
        try:
            return cv2.detail_GraphCutSeamFinderGC("COST_COLOR")
        except Exception:
            pass
    # fallback → DP (COLOR)
    try:
        return cv2.detail_DpSeamFinder("COLOR")
    except Exception:
        # último recurso: Voronoi (menos preciso)
        return cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)

def apply_exposure_comp(corners, images, masks):
    # GAIN_BLOCKS costuma ser estável
    comp = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    comp.feed(corners, images, masks)
    for i in range(len(images)):
        comp.apply(i, corners[i], images[i], masks[i])

def multiband_blend_pair(corner, size, images, masks, num_bands=5):
    blender = cv2.detail_MultiBandBlender()
    blender.setNumBands(int(num_bands))
    blender.prepare(corner, size)
    for img, msk, cor in zip(images, masks, [corner]*len(images)):
        blender.feed(img.astype(np.int16), msk, cor)
    result, result_mask = blender.blend(None, None)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# ---------- pipeline por par (canvas + imagem atual) ----------
def stitch_step(canvas, cur, seam_kind="dp", num_bands=5):
    h, w = cur.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)

    # assumimos que H já mapeia cur → canvas
    # (warp e bbox calculados fora)
    # esta função só cuida de: compensação, seam e multiband no par base/warp
    raise NotImplementedError  # placeholder; mantemos implementação integrada abaixo

# ---------- pipeline completo ----------
def stitch_seams(imgs: List[np.ndarray], detector_name: str, matcher_name: str, ratio: float, ransac: float,
                 conf: float, min_side: int, maxIters: int, seam: str, bands: int):
    det = build_detector(detector_name)
    mat = build_matcher(matcher_name, detector_name)

    canvas = resize_min_side(imgs[0], min_side)

    for i in range(1, len(imgs)):
        cur = resize_min_side(imgs[i], min_side)

        # 1) estimar homografia cur → canvas
        H, inl, good = find_h(cur, canvas, det, mat, ratio, ransac, conf, maxIters)
        if H is None:
            raise SystemExit(f"homografia insuficiente na imagem {i} (matches={len(good)})")

        # 2) calcular bounds do mosaico expandido
        h, w = cur.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        wc = cv2.perspectiveTransform(corners, H)
        cc = np.vstack((wc, np.float32([[0,0],[canvas.shape[1],0],[canvas.shape[1],canvas.shape[0]],[0,canvas.shape[0]]]).reshape(-1,1,2)))
        [xmin,ymin] = np.int32(cc.min(axis=0).ravel()-0.5)
        [xmax,ymax] = np.int32(cc.max(axis=0).ravel()+0.5)
        T = np.array([[1,0,-xmin],[0,1,-ymin],[0,0,1]], dtype=np.float32)

        # 3) warpar atual e posicionar canvas na área comum
        warp = cv2.warpPerspective(cur, T@H, (xmax-xmin, ymax-ymin))
        base = np.zeros_like(warp)
        base[-ymin:-ymin+canvas.shape[0], -xmin:-xmin+canvas.shape[1]] = canvas

        # 4) máscaras iniciais
        warp_mask = (cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)>0).astype(np.uint8) * 255
        base_mask = (cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)>0).astype(np.uint8) * 255

        # 5) compensação de exposição (GAIN_BLOCKS) nos dois frames
        corners_list = [(0,0), (0,0)]  # ambas imagens começam no canto (já no mesmo canvas)
        imgs_list = [base.astype(np.float32), warp.astype(np.float32)]
        masks_list = [base_mask.copy(), warp_mask.copy()]
        apply_exposure_comp(corners_list, imgs_list, masks_list)
        base_comp = imgs_list[0].astype(np.uint8)
        warp_comp = imgs_list[1].astype(np.uint8)

        # 6) seam finding (refina máscaras nas regiões de overlap)
        seam_finder = make_seam_finder(seam)
        imgs_f = [base_comp.astype(np.float32), warp_comp.astype(np.float32)]
        corners_s = [(0,0), (0,0)]
        masks_s = [base_mask.copy(), warp_mask.copy()]
        seam_finder.find(imgs_f, corners_s, masks_s)
        base_mask_ref, warp_mask_ref = masks_s

        # 7) multiband blending no retângulo total
        corner = (0,0)
        size = (xmax-xmin, ymax-ymin)
        result = multiband_blend_pair(corner, size,
                                      [base_comp, warp_comp],
                                      [base_mask_ref, warp_mask_ref],
                                      num_bands=bands)

        canvas = result

    return canvas

# ---------- CLI ----------
def cli_main():
    p = argparse.ArgumentParser(prog="classic.seams_exposure_multiband")
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

    p.add_argument("--seam", choices=["dp","graphcut"], default="dp", help="método de corte de costura")
    p.add_argument("--bands", type=int, default=5, help="nº de bandas do multiband blender")
    p.add_argument("--out", default="pano_seams.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    pano = stitch_seams(imgs, a.detector, a.matcher, a.ratio, a.ransac, a.conf,
                        a.min_side, a.maxIters, a.seam, a.bands)
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
