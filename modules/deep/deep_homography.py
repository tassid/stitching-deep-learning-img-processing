# modules/deep/deep_homography.py
"""
Deep Homography (H4pt/8pt regression) + stitching
- CNN recebe par de imagens (concatenadas em canais) e estima offsets dos 4 cantos (H4pt = 8 valores).
- A homografia é obtida via cv2.getPerspectiveTransform entre cantos originais e deslocados.
- Pipeline aplica estimativa par-a-par e compõe até a imagem base, com feather blending.

Uso (local):
  python -m modules.deep.deep_homography --imgs a.jpg b.jpg c.jpg --weights weights/h4pt.pth --min_side 960 --out pano_deeph.jpg

Uso (Hugging Face):
  python -m modules.deep.deep_homography --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
      --count 24 --offset 300 --weights weights/h4pt.pth --min_side 960 --out pano_deeph.jpg

Fallback (se não tiver pesos ainda):
  python -m modules.deep.deep_homography --imgs a.jpg b.jpg c.jpg --fallback sift --out pano_fallback.jpg

Requisitos:
  pip install opencv-contrib-python torch datasets pillow "huggingface_hub[hf_xet]"
"""

from __future__ import annotations
import argparse, sys
from typing import List, Tuple, Optional
import numpy as np
import cv2

# ----------------- utils de IO -----------------
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

def resize_min_side(img: np.ndarray, min_side: int):
    if min_side <= 0: return img
    h,w = img.shape[:2]; s = min(h,w)
    if s <= min_side: return img
    scale = min_side / float(s)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# ----------------- Modelo (H4pt) -----------------
class HomographyNet:
    """CNN simples para regressão H4pt (8 offsets) inspirado em DeTone et al., 2016 (HomographyNet)."""
    def __init__(self, device: str = "cpu"):
        import torch, torch.nn as nn
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/4
            nn.Conv2d(128,256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 1/8
            nn.Conv2d(256,256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Flatten(),
            nn.Linear(256*8*8, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 8)  # dx1,dy1, dx2,dy2, dx3,dy3, dx4,dy4
        ).to(device)
        self.torch = __import__("torch")

    def load(self, path: str):
        st = self.torch.load(path, map_location=self.device)
        self.model.load_state_dict(st, strict=False)
        self.model.eval()

    def predict_offsets(self, im0: np.ndarray, im1: np.ndarray, input_size=(240,320)) -> np.ndarray:
        """Retorna offsets (8,) em pixels relativos à resolução input_size (h,w)."""
        import torch
        h_in, w_in = input_size
        def prep(im):
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (w_in, h_in), interpolation=cv2.INTER_AREA)
            t = (g.astype(np.float32)/255.0)[None,None,:,:]  # (1,1,H,W)
            return t
        t0 = prep(im0); t1 = prep(im1)
        x = np.concatenate([t0,t1], axis=1)
        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            y = self.model(x).cpu().numpy().reshape(-1)
        return y  # (8,)

# ----------------- Estimar H a partir de offsets -----------------
def homography_from_offsets(img_shape: Tuple[int,int], offsets: np.ndarray, input_size=(240,320)) -> np.ndarray:
    """Mapeia imagem A -> B usando offsets estimados nos 4 cantos de A (ordem TL, TR, BR, BL)."""
    h, w = img_shape[:2]
    h_in, w_in = input_size
    # cantos em resolução de entrada
    src = np.float32([[0,0],[w_in-1,0],[w_in-1,h_in-1],[0,h_in-1]])
    off = offsets.reshape(4,2).astype(np.float32)
    dst = src + off
    # remapear cantos para resolução original
    sx = w / float(w_in); sy = h / float(h_in)
    src_full = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dst_full = np.float32([[dst[0,0]*sx, dst[0,1]*sy],
                           [dst[1,0]*sx, dst[1,1]*sy],
                           [dst[2,0]*sx, dst[2,1]*sy],
                           [dst[3,0]*sx, dst[3,1]*sy]])
    H = cv2.getPerspectiveTransform(src_full, dst_full)
    return H

# ----------------- Fallback clássico -----------------
def sift_homography(a: np.ndarray, b: np.ndarray, ratio=0.75) -> Optional[np.ndarray]:
    sift = cv2.SIFT_create()
    ka,da = sift.detectAndCompute(a,None); kb,db = sift.detectAndCompute(b,None)
    if da is None or db is None: return None
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn=matcher.knnMatch(da,db,k=2)
    good=[m for m,n in knn if m.distance < n.distance*ratio]
    if len(good) < 4: return None
    src=np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst=np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H,_=cv2.findHomography(src,dst, cv2.USAC_MAGSAC, 3.0, maxIters=10000, confidence=0.999)
    return H

# ----------------- Warp/blend -----------------
def warp_all_to_canvas(imgs: List[np.ndarray], Hs: List[np.ndarray]):
    corners_all = []
    for im, H in zip(imgs, Hs):
        h,w = im.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        wc = cv2.perspectiveTransform(corners, H)
        corners_all.append(wc)
    cc = np.vstack(corners_all)
    xmin,ymin = np.floor(cc.min(axis=0).ravel()).astype(np.int32)
    xmax,ymax = np.ceil (cc.max(axis=0).ravel()).astype(np.int32)
    Tx, Ty = -xmin, -ymin
    T = np.array([[1,0,Tx],[0,1,Ty],[0,0,1]], dtype=np.float32)
    size = (int(xmax - xmin), int(ymax - ymin))
    warps, masks = [], []
    for im, H in zip(imgs, Hs):
        W = T @ H
        wim = cv2.warpPerspective(im, W, size)
        mask = (cv2.cvtColor(wim, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)*255
        warps.append(wim); masks.append(mask)
    return warps, masks

def feather_blend_pair(base: np.ndarray, warp: np.ndarray, mask_base: np.ndarray, mask_warp: np.ndarray) -> np.ndarray:
    k = np.ones((3,3), np.uint8)
    mask_base = cv2.morphologyEx(mask_base, cv2.MORPH_DILATE, k)
    mask_warp = cv2.morphologyEx(mask_warp, cv2.MORPH_DILATE, k)
    dt_b = cv2.distanceTransform((mask_base>0).astype(np.uint8), cv2.DIST_L2, 5)
    dt_w = cv2.distanceTransform((mask_warp>0).astype(np.uint8), cv2.DIST_L2, 5)
    s = dt_b + dt_w + 1e-6
    wb = (dt_b/s)[...,None]; ww = (dt_w/s)[...,None]
    out = base.astype(np.float32)*wb + warp.astype(np.float32)*ww
    return np.clip(out,0,255).astype(np.uint8)

def blend_sequence(warps: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    canvas = warps[0].copy()
    m_base = masks[0].copy()
    for i in range(1, len(warps)):
        canvas = feather_blend_pair(canvas, warps[i], m_base, masks[i])
        m_base = ((cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)>0).astype(np.uint8))*255
    return canvas

# ----------------- Pipeline -----------------
def stitch_deep_homography(imgs: List[np.ndarray], weights: Optional[str], device: str = "cpu",
                           min_side: int = 960, ransac_thresh: float = 3.0,
                           fallback: str = "sift") -> np.ndarray:
    imgs_r = [resize_min_side(im, min_side) for im in imgs]
    Hs = [np.eye(3, dtype=np.float32)]
    use_deep = weights is not None and len(str(weights))>0
    if use_deep:
        try:
            net = HomographyNet(device=device)
            net.load(weights)
        except Exception as e:
            print(f"[aviso] não consegui carregar pesos deep ({e}); usando fallback='{fallback}'")
            use_deep = False

    for i in range(1, len(imgs_r)):
        if use_deep:
            offsets = net.predict_offsets(imgs_r[i], imgs_r[i-1])  # i -> i-1
            H = homography_from_offsets(imgs_r[i].shape, offsets)
        else:
            if fallback == "sift":
                H = sift_homography(imgs_r[i], imgs_r[i-1])
            else:
                H = None
        if H is None:
            raise SystemExit(f"homografia insuficiente no par {i-1}->{i}")
        Hs.append((Hs[-1] @ H).astype(np.float32))

    warps, masks = warp_all_to_canvas(imgs_r, Hs)
    pano = blend_sequence(warps, masks)
    return pano

# ----------------- CLI -----------------
def cli_main():
    p = argparse.ArgumentParser(prog="deep.deep_homography")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--weights", help="arquivo .pth da rede (H4pt). Se ausente, usa fallback.")
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--min_side", type=int, default=960)
    p.add_argument("--ransac_thresh", type=float, default=3.0)
    p.add_argument("--fallback", choices=["sift","none"], default="sift")
    p.add_argument("--out", default="pano_deeph.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)

    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    pano = stitch_deep_homography(imgs, weights=a.weights, device=a.device,
                                  min_side=a.min_side, ransac_thresh=a.ransac_thresh,
                                  fallback=a.fallback)
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
