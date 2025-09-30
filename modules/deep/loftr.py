# -*- coding: utf-8 -*-
# modules/deep/loftr.py
"""
LoFTR-based stitching (deep feature matching, detector-free).
Requisitos: pip install opencv-contrib-python torch kornia datasets pillow
"""

from __future__ import annotations
import argparse, sys, os
from typing import List, Tuple, Optional
import numpy as np
import cv2

__all__ = ["stitch", "stitch_loftr"]

# ----------------- Utils -----------------
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
                im = cv2.imread(img, cv2.IMREAD_UNCHANGED)
                if im is not None:
                    imgs.append(im)
    if len(imgs) < 2:
        sys.exit("Não consegui ler imagens RGB desse dataset/split.")
    return imgs

def ensure_gray_8u(img: np.ndarray) -> np.ndarray:
    """Converte qualquer entrada (BGR/BGRA/GRAY/GRAY+ALPHA, 8/16 bits) para GRAY 8-bit (H,W)."""
    if img is None:
        raise ValueError("Imagem None.")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        c = img.shape[2]
        if c == 1:   gray = img[..., 0]
        elif c == 2: gray = img[..., 0]  # descarta alpha/canal extra
        elif c == 3: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif c == 4: gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:        gray = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Dimensão inesperada: {img.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray

# ----------------- Visualização (opcional) -----------------
def save_match_vis(img0: np.ndarray, img1: np.ndarray, k0: np.ndarray, k1: np.ndarray, path: str, max_draw: int = 500):
    vis0 = img0.copy(); vis1 = img1.copy()
    h = max(vis0.shape[0], vis1.shape[0]); w = vis0.shape[1] + vis1.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:vis0.shape[0], :vis0.shape[1]] = vis0
    canvas[:vis1.shape[0], vis0.shape[1]:] = vis1
    step = max(1, int(len(k0) / max_draw))
    for i in range(0, len(k0), step):
        p0 = tuple(np.round(k0[i]).astype(int))
        p1 = tuple(np.round(k1[i]).astype(int))
        p1_shift = (p1[0] + vis0.shape[1], p1[1])
        cv2.line(canvas, p0, p1_shift, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, (0, 0, 255), -1)
        cv2.circle(canvas, p1_shift, 2, (255, 0, 0), -1)
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    cv2.imwrite(path, canvas)

# ----------------- LoFTR Matching -----------------
def loftr_match(im0: np.ndarray, im1: np.ndarray, device: str = "cpu", weights: str = "outdoor",
                conf_thresh: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from kornia.feature import LoFTR

    if weights not in {"outdoor", "indoor"}:
        weights = "outdoor"
    matcher = LoFTR(pretrained=weights).to(device).eval()

    def to_tensor_gray(img_any: np.ndarray) -> "torch.Tensor":
        # Versão independente do Kornia: garante (1,1,H,W)
        gray = ensure_gray_8u(img_any)               # (H,W), uint8
        t = torch.from_numpy(gray).contiguous().float() / 255.0  # (H,W)
        t = t.unsqueeze(0).unsqueeze(0)              # (1,1,H,W)
        return t

    t0 = to_tensor_gray(im0)
    t1 = to_tensor_gray(im1)

    # Sanidade: B=1, C=1
    if not (t0.ndim == 4 and t0.shape[1] == 1):
        raise RuntimeError(f"Esperado (B,1,H,W), recebi {tuple(t0.shape)}")
    if not (t1.ndim == 4 and t1.shape[1] == 1):
        raise RuntimeError(f"Esperado (B,1,H,W), recebi {tuple(t1.shape)}")

    with torch.no_grad():
        data = {"image0": t0.to(device), "image1": t1.to(device)}
        out = matcher(data)
        k0 = out["keypoints0"].detach().cpu().numpy()
        k1 = out["keypoints1"].detach().cpu().numpy()
        conf = out["confidence"].detach().cpu().numpy()

    if conf_thresh is not None and conf_thresh > 0:
        m = conf >= conf_thresh
        k0, k1, conf = k0[m], k1[m], conf[m]
    return k0, k1, conf

def estimate_homography(k0: np.ndarray, k1: np.ndarray, ransac_thresh: float = 3.0) -> Optional[np.ndarray]:
    if k0.shape[0] < 4 or k1.shape[0] < 4:
        return None
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, _ = cv2.findHomography(k0, k1, method, ransac_thresh, maxIters=10000, confidence=0.999)
    return H

# ----------------- Stitching helpers -----------------
def resize_min_side(img: np.ndarray, min_side: int):
    if min_side <= 0:
        return img
    h, w = img.shape[:2]; s = min(h, w)
    if s <= min_side:
        return img
    scale = min_side / float(s)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def warp_all_to_canvas(imgs: List[np.ndarray], Hs: List[np.ndarray]):
    corners_all = []
    for im, H in zip(imgs, Hs):
        h, w = im.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        wc = cv2.perspectiveTransform(corners, H)
        corners_all.append(wc)
    cc = np.vstack(corners_all)
    xmin, ymin = np.floor(cc.min(axis=0).ravel()).astype(np.int32)
    xmax, ymax = np.ceil (cc.max(axis=0).ravel()).astype(np.int32)
    Tx, Ty = -xmin, -ymin
    T = np.array([[1,0,Tx],[0,1,Ty],[0,0,1]], dtype=np.float32)
    size = (int(xmax - xmin), int(ymax - ymin))
    warps, masks = [], []
    for im, H in zip(imgs, Hs):
        W = T @ H
        wim = cv2.warpPerspective(im, W, size)
        mask = (cv2.cvtColor(wim, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
        warps.append(wim); masks.append(mask)
    return warps, masks, size, T

def feather_blend_pair(base: np.ndarray, warp: np.ndarray, mask_base: np.ndarray, mask_warp: np.ndarray) -> np.ndarray:
    k = np.ones((3,3), np.uint8)
    mask_base = cv2.morphologyEx(mask_base, cv2.MORPH_DILATE, k)
    mask_warp  = cv2.morphologyEx(mask_warp,  cv2.MORPH_DILATE, k)
    dt_b = cv2.distanceTransform((mask_base>0).astype(np.uint8), cv2.DIST_L2, 5)
    dt_w = cv2.distanceTransform((mask_warp >0).astype(np.uint8), cv2.DIST_L2, 5)
    s = dt_b + dt_w + 1e-6
    wb = (dt_b/s)[...,None]; ww = (dt_w/s)[...,None]
    out = base.astype(np.float32)*wb + warp.astype(np.float32)*ww
    return np.clip(out,0,255).astype(np.uint8)

def blend_sequence(warps: List[np.ndarray], masks: List[np.ndarray]) -> np.ndarray:
    canvas = warps[0].copy()
    m_base = masks[0].copy()
    for i in range(1, len(warps)):
        canvas = feather_blend_pair(canvas, warps[i], m_base, masks[i])
        m_base = ((cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)>0).astype(np.uint8)) * 255
    return canvas

# ----------------- Pipeline -----------------
def stitch_loftr(imgs: List[np.ndarray], device: str = "cpu", weights: str = "outdoor",
                 min_side: int = 960, conf_thresh: float = 0.2, ransac_thresh: float = 3.0,
                 save_matches_dir: Optional[str] = None) -> np.ndarray:
    imgs_r = [resize_min_side(im, min_side) for im in imgs]
    Hs = [np.eye(3, dtype=np.float32)]
    for i in range(1, len(imgs_r)):
        k0, k1, cf = loftr_match(imgs_r[i], imgs_r[i-1], device=device, weights=weights, conf_thresh=conf_thresh)
        if save_matches_dir:
            sp = f"{save_matches_dir}/match_{i-1}_{i}.jpg"
            save_match_vis(imgs_r[i], imgs_r[i-1], k0, k1, sp)
        H = estimate_homography(k0, k1, ransac_thresh=ransac_thresh)
        if H is None:
            raise SystemExit(f"homografia insuficiente no par {i-1}->{i} (matches={k0.shape[0]})")
        Hs.append((Hs[-1] @ H).astype(np.float32))
    warps, masks, size, T = warp_all_to_canvas(imgs_r, Hs)
    pano = blend_sequence(warps, masks)
    return pano

# --- API para GUI ---
def stitch(imgs: List[np.ndarray], device: str = "cpu", weights: str = "outdoor",
           min_side: int = 960, conf_thresh: float = 0.2, ransac_thresh: float = 3.0) -> np.ndarray:
    return stitch_loftr(imgs, device=device, weights=weights, min_side=min_side,
                        conf_thresh=conf_thresh, ransac_thresh=ransac_thresh)

# ----------------- CLI -----------------
def cli_main():
    p = argparse.ArgumentParser(prog="deep.loftr")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--device", choices=["cpu","cuda","mps"], default="cpu")
    p.add_argument("--weights", choices=["outdoor","indoor"], default="outdoor")
    p.add_argument("--min_side", type=int, default=960)
    p.add_argument("--conf_thresh", type=float, default=0.2)
    p.add_argument("--ransac_thresh", type=float, default=3.0)
    p.add_argument("--save_matches_dir", default=None)
    p.add_argument("--out", default="pano_loftr.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x, cv2.IMREAD_UNCHANGED) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)

    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    if a.save_matches_dir:
        os.makedirs(a.save_matches_dir, exist_ok=True)

    pano = stitch_loftr(imgs, device=a.device, weights=a.weights, min_side=a.min_side,
                        conf_thresh=a.conf_thresh, ransac_thresh=a.ransac_thresh,
                        save_matches_dir=a.save_matches_dir)
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
