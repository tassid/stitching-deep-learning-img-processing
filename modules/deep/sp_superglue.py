# modules/deep/sp_superglue.py
"""
DISK + LightGlue stitching (Kornia 0.8.1+)
- Detecta keypoints e descritores com DISK (Deep Image Structure Keypoints)
- Faz matching robusto com LightGlue
- Estima homografia (USAC_MAGSAC) e faz warping incremental + feather blending

Nota: Substituído SuperPoint+SuperGlue por DISK+LightGlue para compatibilidade com Kornia 0.8.1+

Uso (arquivos locais):
  python -m modules.deep.sp_superglue --imgs a.jpg b.jpg c.jpg --min_side 960 --out pano_disk_lg.jpg

Uso (Hugging Face):
  python -m modules.deep.sp_superglue --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train --count 24 --offset 300 --min_side 960 --out pano_disk_lg.jpg

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

# --------------- DISK + LightGlue ---------------
def sp_sg_match(im0: np.ndarray, im1: np.ndarray, device: str = "cpu",
                sp_weights: str = "synthetic", sg_weights: str = "outdoor",
                keypoint_thresh: float = 0.005, max_kp: int = 4096,
                match_conf_thresh: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Usa DISK + LightGlue (disponível no Kornia 0.8.1+)
    Nota: sp_weights e sg_weights são ignorados (mantidos por compatibilidade)
    """
    import torch
    import kornia as K
    from kornia.feature import DISK, LightGlueMatcher
    from kornia.feature.laf import laf_from_center_scale_ori

    # DISK feature extractor
    disk = DISK.from_pretrained("depth").to(device).eval()
    # LightGlue matcher
    matcher = LightGlueMatcher("disk").to(device).eval()

    def to_tensor_rgb(img_bgr: np.ndarray) -> "torch.Tensor":
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = K.image_to_tensor(rgb, keepdim=False).float() / 255.0   # (1,3,H,W)
        return t

    t0 = to_tensor_rgb(im0).to(device)
    t1 = to_tensor_rgb(im1).to(device)

    with torch.no_grad():
        # Extract features with DISK
        feats0_list = disk(t0, max_kp)
        feats1_list = disk(t1, max_kp)

        # Validar se detectou features
        if len(feats0_list) == 0 or len(feats1_list) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

        feats0 = feats0_list[0]  # DISKFeatures object
        feats1 = feats1_list[0]

        # Get keypoints and descriptors
        kpts0 = feats0.keypoints  # (N, 2)
        kpts1 = feats1.keypoints  # (M, 2)
        desc0 = feats0.descriptors  # (N, 128)
        desc1 = feats1.descriptors  # (M, 128)

        # Create LAFs from keypoints (center, scale=1, orientation=0)
        n0 = kpts0.shape[0]
        n1 = kpts1.shape[0]

        # Verificar se há keypoints
        if n0 == 0 or n1 == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)
        lafs0 = laf_from_center_scale_ori(
            kpts0.unsqueeze(0),  # (1, N, 2)
            torch.ones(1, n0, 1, 1, device=device),  # scale
            torch.zeros(1, n0, 1, device=device)  # orientation
        )  # (1, N, 2, 3)
        lafs1 = laf_from_center_scale_ori(
            kpts1.unsqueeze(0),
            torch.ones(1, n1, 1, 1, device=device),
            torch.zeros(1, n1, 1, device=device)
        )

        # Match with LightGlue
        dists, matches = matcher(
            desc0.unsqueeze(0), desc1.unsqueeze(0),  # (1, N, 128), (1, M, 128)
            lafs0, lafs1,
            hw1=tuple(t0.shape[-2:]), hw2=tuple(t1.shape[-2:])
        )

        # matches: (1, N) com índices válidos ou -1
        matches = matches[0].cpu().numpy()  # (N,)
        dists = dists[0].cpu().numpy()  # (N,)

        # Convert distances to scores (lower distance = higher score)
        scores = 1.0 / (1.0 + dists)

    # Filter valid matches
    valid = matches > -1
    if match_conf_thresh is not None and match_conf_thresh > 0:
        valid = np.logical_and(valid, scores >= match_conf_thresh)

    idx0 = np.where(valid)[0]

    # Check if we have any matches
    if len(idx0) == 0:
        # Return empty arrays
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

    idx1 = matches[valid].astype(int)

    mk0 = kpts0[idx0].cpu().numpy()
    mk1 = kpts1[idx1].cpu().numpy()
    mconf = scores[idx0]

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

        num_matches = k0.shape[0]
        if num_matches == 0:
            raise RuntimeError(f"Nenhum match encontrado entre imagens {i-1} e {i}. Tente: 1) Reduzir match_conf_thresh (atual: {match_conf_thresh}), 2) Aumentar max_kp (atual: {max_kp}), 3) Usar imagens com mais overlap")

        if save_matches_dir:
            sp = f"{save_matches_dir}/match_{i-1}_{i}.jpg"
            save_match_vis(imgs_r[i], imgs_r[i-1], k0, k1, sp)

        H = estimate_homography(k0, k1, ransac_thresh=ransac_thresh)
        if H is None:
            raise RuntimeError(f"Homografia insuficiente no par {i-1}->{i}. Matches encontrados: {num_matches}, mas <4 inliers após RANSAC. Tente aumentar ransac_thresh ou verificar overlap entre imagens.")
        Hs.append((Hs[-1] @ H).astype(np.float32))
    warps, masks, size, T = warp_all_to_canvas(imgs_r, Hs)
    pano = blend_sequence(warps, masks)
    return pano

# --- API para GUI ---
def stitch(imgs: List[np.ndarray], device: str = "cpu", try_gpu: bool = False) -> np.ndarray:
    """Interface simplificada para a GUI."""
    if try_gpu and device == "cpu":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
        except Exception:
            pass
    # Usar parâmetros mais permissivos para a GUI
    return stitch_sp_sg(imgs, device=device, min_side=960, max_kp=2048, match_conf_thresh=0.0, ransac_thresh=5.0)

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

    pano = stitch_sp_sg(imgs, device=a.device, sp_weights=a.sp_weights,
                        sg_weights=a.sg_weights,
                        min_side=a.min_side, keypoint_thresh=a.keypoint_thresh, max_kp=a.max_kp,
                        match_conf_thresh=a.match_conf_thresh, ransac_thresh=a.ransac_thresh,
                        save_matches_dir=a.save_matches_dir)
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
