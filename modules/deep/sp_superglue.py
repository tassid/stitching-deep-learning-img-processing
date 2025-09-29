# modules/deep/sp_superglue.py
"""
SuperPoint + SuperGlue stitching
- Detecta keypoints e descritores com SuperPoint (Kornia)
- Faz matching robusto com SuperGlue (Kornia)
- Estima homografia (USAC_MAGSAC) e faz warping incremental + feather blending

Uso (arquivos locais):
  python -m modules.deep.sp_superglue --imgs a.jpg b.jpg c.jpg --sp-weights synthetic --sg-weights outdoor --min_side 960 --out pano_sp_sg.jpg

Uso (Hugging Face):
  python -m modules.deep.sp_superglue --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train --count 24 --offset 300 --min_side 960 --out pano_sp_sg.jpg

Requisitos:
  pip install opencv-contrib-python torch kornia datasets pillow "huggingface_hub[hf_xet]"
"""

from __future__ import annotations
import argparse, sys
from typing import List, Tuple, Optional
import numpy as np
import cv2

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

# --------------- SuperPoint + SuperGlue ---------------
def sp_sg_match(im0: np.ndarray, im1: np.ndarray, device: str = "cpu",
                sp_weights: str = "synthetic", sg_weights: str = "outdoor",
                keypoint_thresh: float = 0.005, max_kp: int = 4096,
                match_conf_thresh: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    import kornia as K
    from kornia.feature import SuperPoint, SuperGlue, KeypointDetector

    # SuperPoint
    sp = SuperPoint(max_num_keypoints=max_kp, detection_threshold=keypoint_thresh, pretrained=sp_weights).to(device).eval()
    # SuperGlue
    sg = SuperGlue(pretrained=sg_weights).to(device).eval()

    def to_tensor_gray(img_bgr: np.ndarray) -> "torch.Tensor":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        t = K.image_to_tensor(gray, keepdim=True).float() / 255.0   # (1,1,H,W)
        return t

    t0 = to_tensor_gray(im0).to(device)
    t1 = to_tensor_gray(im1).to(device)

    with torch.no_grad():
        lafs0, desc0 = sp(t0)  # lafs: (1,N,2,3), desc: (1,N,256)
        lafs1, desc1 = sp(t1)
        kpts0 = lafs0[:, :, :, -1].squeeze(0).cpu().numpy()  # (N,2)
        kpts1 = lafs1[:, :, :, -1].squeeze(0).cpu().numpy()

        data = {
            "descriptors0": desc0,
            "descriptors1": desc1,
            "keypoints0": lafs0,
            "keypoints1": lafs1,
        }
        out = sg(data)

        # matches0: para cada kp0, índice do kp1 correspondente (ou -1)
        matches0 = out["matches0"][0].detach().cpu().numpy()
        conf = out["matching_scores0"][0].detach().cpu().numpy()

    # coletar pares válidos
    valid = matches0 > -1
    if match_conf_thresh is not None and match_conf_thresh > 0:
        valid = np.logical_and(valid, conf >= match_conf_thresh)

    idx0 = np.where(valid)[0]
    idx1 = matches0[valid].astype(int)

    mk0 = kpts0[idx0]
    mk1 = kpts1[idx1]
    mconf = conf[idx0]

    return mk0, mk1, mconf

def estimate_homography(k0: np.ndarray, k1: np.ndarray, ransac_thresh: float = 3.0) -> Optional[np.ndarray]:
    if k0.shape[0] < 4 or k1.shape[0] < 4:
        return None
    H, mask = cv2.findHomography(k0, k1, cv2.USAC_MAGSAC, ransac_thresh, maxIters=10000, confidence=0.999)
    return H

# --------------- Stitching helpers ---------------
def resize_min_side(img: np.ndarray, min_side: int):
    if min_side <= 0: return img
    h,w = img.shape[:2]; s = min(h,w)
    if s <= min_side: return img
    scale = min_side / float(s)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

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
    return warps, masks, size, T

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

# --------------- Visualization ---------------
def save_match_vis(im_a, im_b, k0, k1, save_path: str, max_lines: int = 2000):
    ha, wa = im_a.shape[:2]; hb, wb = im_b.shape[:2]
    H = max(ha, hb); W = wa + wb
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[:ha,:wa] = im_a
    vis[:hb,wa:wa+wb] = im_b
    n = k0.shape[0]
    idx = np.arange(n)
    if n > max_lines:
        idx = np.random.choice(n, size=max_lines, replace=False)
    for i in idx:
        x0,y0 = k0[i]
        x1,y1 = k1[i]
        color = tuple(np.random.randint(0,255, size=3).tolist())
        cv2.circle(vis, (int(x0), int(y0)), 2, color, -1)
        cv2.circle(vis, (int(x1)+wa, int(y1)), 2, color, -1)
        cv2.line(vis, (int(x0), int(y0)), (int(x1)+wa, int(y1)), color, 1)
    cv2.imwrite(save_path, vis)

# --------------- Pipeline ---------------
def stitch_sp_sg(imgs: List[np.ndarray], device: str = "cpu",
                 sp_weights: str = "synthetic", sg_weights: str = "outdoor",
                 min_side: int = 960, keypoint_thresh: float = 0.005, max_kp: int = 4096,
                 match_conf_thresh: float = 0.2, ransac_thresh: float = 3.0,
                 save_matches_dir: Optional[str] = None) -> np.ndarray:
    imgs_r = [resize_min_side(im, min_side) for im in imgs]
    Hs = [np.eye(3, dtype=np.float32)]
    for i in range(1, len(imgs_r)):
        k0, k1, conf = sp_sg_match(imgs_r[i], imgs_r[i-1], device=device,
                                   sp_weights=sp_weights, sg_weights=sg_weights,
                                   keypoint_thresh=keypoint_thresh, max_kp=max_kp,
                                   match_conf_thresh=match_conf_thresh)
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

# --------------- CLI ---------------
def cli_main():
    p = argparse.ArgumentParser(prog="deep.sp_superglue")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--sp-weights", choices=["synthetic"], default="synthetic")
    p.add_argument("--sg-weights", choices=["outdoor","indoor"], default="outdoor")
    p.add_argument("--min_side", type=int, default=960)
    p.add_argument("--keypoint_thresh", type=float, default=0.005)
    p.add_argument("--max_kp", type=int, default=4096)
    p.add_argument("--match_conf_thresh", type=float, default=0.2)
    p.add_argument("--ransac_thresh", type=float, default=3.0)
    p.add_argument("--save_matches_dir", default=None)
    p.add_argument("--out", default="pano_sp_sg.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)

    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    if a.save_matches_dir:
        import os
        os.makedirs(a.save_matches_dir, exist_ok=True)

    pano = stitch_sp_sg(imgs, device=a.device, sp_weights=a["sp_weights"] if isinstance(a,dict) else a.sp_weights,
                        sg_weights=a["sg_weights"] if isinstance(a,dict) else a.sg_weights,
                        min_side=a.min_side, keypoint_thresh=a.keypoint_thresh, max_kp=a.max_kp,
                        match_conf_thresh=a.match_conf_thresh, ransac_thresh=a.ransac_thresh,
                        save_matches_dir=a.save_matches_dir)
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
