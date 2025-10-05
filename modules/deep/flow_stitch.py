# modules/deep/flow_stitch.py
"""
Optical-Flow-based Stitching (RAFT / PWC-Net) + RANSAC Homography fit
- Estima fluxo denso entre pares consecutivos
- Converte fluxo (x,y)->(x+u,y+v) em correspondências e ajusta H via USAC_MAGSAC
- Faz warping incremental para um canvas global e blending por feather distance

Uso (local):
  python -m modules.deep.flow_stitch --imgs a.jpg b.jpg c.jpg --model raft --min_side 960 --out pano_flow.jpg

Uso (Hugging Face):
  python -m modules.deep.flow_stitch --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
     --count 24 --offset 300 --model raft --min_side 960 --out pano_flow.jpg

Requisitos:
  pip install opencv-contrib-python torch torchvision kornia datasets pillow "huggingface_hub[hf_xet]"
  # RAFT (torch.hub) baixa pesos da internet na 1ª vez: 'princeton-vl/RAFT'
  # PWC-Net (opcional) pode exigir repositório compatível (ex.: 'sniklaus/pytorch-pwc') — uso experimental.

Notas:
- Se torch.hub não conseguir baixar, você pode apontar para um checkpoint local e adaptar a função load_model().
"""

from __future__ import annotations
import argparse, sys, math, os
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

def resize_min_side(img: np.ndarray, min_side: int):
    if min_side <= 0: return img
    h,w = img.shape[:2]; s = min(h,w)
    if s <= min_side: return img
    scale = min_side / float(s)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

# -------- Optical Flow loaders --------
def load_model(model_name: str = "raft", device: str = "cpu"):
    import torch
    model_name = model_name.lower()
    if model_name == "raft":
        # RAFT via torch.hub (princeton-vl/RAFT). Precisa internet na 1ª vez
        try:
            raft = torch.hub.load('princeton-vl/RAFT', 'raft_large', pretrained=True, source='github', force_reload=False, trust_repo=True)
            raft = raft.to(device).eval()
            return ("raft", raft)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar RAFT via torch.hub: {e}\n\nDica: Flow Stitch requer download do modelo RAFT (primeira vez). Verifique conexão com internet.")
    elif model_name == "pwc":
        try:
            # Exemplo usando 'sniklaus/pytorch-pwc' (interface difere; código abaixo é ilustrativo).
            pwc = torch.hub.load('sniklaus/pytorch-pwc', 'pwcnet', pretrained=True)
            pwc = pwc.to(device).eval()
            return ("pwc", pwc)
        except Exception as e:
            raise RuntimeError(f"Falha ao carregar PWC-Net via torch.hub: {e}")
    else:
        raise ValueError("modelo inválido: use raft|pwc")

def flow_raft(model, im0: np.ndarray, im1: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Retorna fluxo denso (H,W,2) em pixels para mapear im0 -> im1."""
    import torch
    import torch.nn.functional as F

    def prep(img):
        # .copy() evita negative stride
        t = torch.from_numpy(img[:, :, ::-1].copy().transpose(2,0,1)).float()[None]/255.0  # BGR->RGB, (1,3,H,W)
        return t.to(device)

    t0 = prep(im0); t1 = prep(im1)
    with torch.no_grad():
        # RAFT geralmente requer pad/divisibilidade; muitos hubs já padronizam internamente
        flow_low, flow_up = model(t0, t1, iters=20, test_mode=True)
        flow = flow_up[0].permute(1,2,0).cpu().numpy()  # (H,W,2)
    return flow

def flow_to_matches(flow: np.ndarray, step: int = 8, conf: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Amostra grelha de pontos do im0 e gera correspondentes no im1 via (x+u,y+v)."""
    H, W = flow.shape[:2]
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    x0 = grid_x.reshape(-1).astype(np.float32)
    y0 = grid_y.reshape(-1).astype(np.float32)
    u = flow[grid_y, grid_x, 0].reshape(-1).astype(np.float32)
    v = flow[grid_y, grid_x, 1].reshape(-1).astype(np.float32)
    k0 = np.stack([x0, y0], axis=1)
    k1 = np.stack([x0 + u, y0 + v], axis=1)
    return k0, k1

def estimate_homography(k0: np.ndarray, k1: np.ndarray, ransac_thresh: float = 3.0) -> Optional[np.ndarray]:
    if k0.shape[0] < 4: return None
    H, mask = cv2.findHomography(k0, k1, cv2.USAC_MAGSAC, ransac_thresh, maxIters=10000, confidence=0.999)
    return H

# -------- Warping and blending --------
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

# -------- Pipeline --------
def stitch_flow(imgs: List[np.ndarray], model_name: str = "raft", device: str = "cpu",
                min_side: int = 960, flow_step: int = 8, ransac_thresh: float = 3.0) -> np.ndarray:
    # resize
    imgs_r = [resize_min_side(im, min_side) for im in imgs]
    # load model
    name, model = load_model(model_name, device)
    Hs = [np.eye(3, dtype=np.float32)]
    for i in range(1, len(imgs_r)):
        if name == "raft":
            flow = flow_raft(model, imgs_r[i], imgs_r[i-1], device=device)
        else:
            # Placeholder para PWC se implementado
            flow = flow_raft(model, imgs_r[i], imgs_r[i-1], device=device)
        k0, k1 = flow_to_matches(flow, step=flow_step)
        H = estimate_homography(k0, k1, ransac_thresh=ransac_thresh)
        if H is None:
            raise SystemExit(f"homografia insuficiente no par {i-1}->{i} (amostras={k0.shape[0]})")
        Hs.append((Hs[-1] @ H).astype(np.float32))
    warps, masks = warp_all_to_canvas(imgs_r, Hs)
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
    return stitch_flow(imgs, model_name="raft", device=device)

# -------- CLI --------
def cli_main():
    p = argparse.ArgumentParser(prog="deep.flow_stitch")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--model", choices=["raft","pwc"], default="raft")
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--min_side", type=int, default=960)
    p.add_argument("--flow_step", type=int, default=8, help="subamostragem da grade de pontos do fluxo")
    p.add_argument("--ransac_thresh", type=float, default=3.0)
    p.add_argument("--out", default="pano_flow.jpg")
    a = p.parse_args()

    if a.imgs:
        imgs = [cv2.imread(x) for x in a.imgs]
    else:
        imgs = load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)

    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    pano = stitch_flow(imgs, model_name=a.model, device=a.device, min_side=a.min_side,
                       flow_step=a.flow_step, ransac_thresh=a.ransac_thresh)
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
