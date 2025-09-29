# modules/deep/harmonize_cnn.py
"""
Photometric Harmonization (CNN) — pós-processamento para panoramas
------------------------------------------------------------------
Rede leve (U-Net simplificado) que aprende um **resíduo RGB** por pixel para
harmonizar exposição/cor do panorama em relação a uma **referência** (p.ex. uma
imagem do conjunto ou uma versão equalizada). Treinada RAPIDAMENTE de forma
auto/semissupervisionada em área de **overlap** (pano vs referência).

Uso (com referência):
  python -m modules.deep.harmonize_cnn --in pano.jpg --ref uma_imagem.jpg --steps 500 --lr 2e-3 --out pano_harmo.jpg

Uso (sem referência — target = equalização LAB/gray-world):
  python -m modules.deep.harmonize_cnn --in pano.jpg --steps 400 --lr 2e-3 --out pano_harmo.jpg

Requisitos:
  pip install opencv-contrib-python torch
"""

from __future__ import annotations
import argparse, sys, os
from typing import Optional, Tuple
import numpy as np
import cv2

# ---------- util ----------
def imread_color(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    return im

def resize_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Redimensiona b -> a (mesma resolução)."""
    ha, wa = a.shape[:2]
    bb = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_AREA)
    return a, bb

def valid_mask(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (g > 0).astype(np.uint8)

def overlap_bbox(a: np.ndarray, b: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ma, mb = valid_mask(a), valid_mask(b)
    m = (ma & mb).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if len(ys) == 0: return None
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    return x0, y0, x1, y1

def gray_world_target(img: np.ndarray) -> np.ndarray:
    """Alvo simples: força médias RGB iguais (gray-world) + leve equalização de L*."""
    tgt = img.astype(np.float32).copy()
    means = tgt.reshape(-1,3).mean(0) + 1e-6
    g = float(means.mean())
    gain = g / means
    tgt *= gain[None,None,:]
    tgt = np.clip(tgt, 0, 255)
    # equalizar leve em L*
    lab = cv2.cvtColor(tgt.astype(np.uint8), cv2.COLOR_BGR2Lab)
    L = lab[...,0]
    lab[...,0] = cv2.equalizeHist(L)
    tgt = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return tgt

# ---------- rede ----------
class ResidualUNet:
    """U-Net reduzida: entrada (3), saída (3) residual em [-1,1] (escala de 0..255)."""
    def __init__(self, device: str = "cpu"):
        import torch, torch.nn as nn
        C=32
        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))
        self.down1 = block(3, C)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = block(C, 2*C)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = block(2*C, 4*C)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = block(4*C+2*C, 2*C)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = block(2*C+C, C)
        self.outc = nn.Conv2d(C, 3, 1)
        self.tanh = nn.Tanh()
        self.device = device
        self.torch = __import__("torch")
        for m in self.modules():
            if isinstance(m, (self.torch.nn.Conv2d,)):
                self.torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def modules(self):
        # helper p/ weight init
        yield from [self.down1, self.down2, self.bridge, self.dec2, self.dec1, self.outc]

    def __call__(self, x):
        import torch
        # encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        br = self.bridge(p2)
        # decoder
        u2 = self.up2(br)
        cat2 = self.torch.cat([u2, d2], dim=1)
        dec2 = self.dec2(cat2)
        u1 = self.up1(dec2)
        cat1 = self.torch.cat([u1, d1], dim=1)
        dec1 = self.dec1(cat1)
        out = self.outc(dec1)
        return self.tanh(out)  # residual em [-1,1]

# ---------- treino rápido ----------
def optimize_harmonizer(img_in: np.ndarray, img_tgt: np.ndarray, mask_overlap: Optional[np.ndarray],
                        steps=400, lr=2e-3, lam_tv=0.01, device="cpu") -> np.ndarray:
    import torch
    net = ResidualUNet(device=device)
    net_t = torch.jit.script if hasattr(torch, "jit") else (lambda x: x)
    # tensores
    x = torch.from_numpy(img_in[:, :, ::-1].transpose(2,0,1)).float()[None]/255.0  # BGR->RGB
    y = torch.from_numpy(img_tgt[:, :, ::-1].transpose(2,0,1)).float()[None]/255.0
    x = x.to(device); y = y.to(device)
    opt = torch.optim.Adam([p for p in net.down1.parameters()] + 
                           [p for p in net.down2.parameters()] + 
                           [p for p in net.bridge.parameters()] + 
                           [p for p in net.dec2.parameters()] + 
                           [p for p in net.dec1.parameters()] + 
                           [p for p in net.outc.parameters()], lr=lr)
    m = None
    if mask_overlap is not None:
        m = torch.from_numpy((mask_overlap>0).astype(np.float32))[None,None].to(device)

    def tv_loss(res):
        dx = res[:,:,:,1:] - res[:,:,:,:-1]
        dy = res[:,:,1:,:] - res[:,:,:-1,:]
        return (dx.abs().mean() + dy.abs().mean())

    for it in range(steps):
        opt.zero_grad()
        res = net(x)                      # [-1,1]
        x_corr = torch.clamp(x + res*0.1, 0.0, 1.0)  # escala pequena p/ estabilidade
        if m is None:
            photometric = (x_corr - y).abs().mean()
        else:
            photometric = ((x_corr - y).abs()*m).sum() / (m.sum() + 1e-6)
        loss = photometric + lam_tv*tv_loss(res)
        loss.backward()
        opt.step()

    # saída
    with torch.no_grad():
        res = net(x)
        x_corr = torch.clamp(x + res*0.1, 0.0, 1.0)
    out = (x_corr[0].permute(1,2,0).cpu().numpy()*255.0).astype(np.uint8)[:, :, ::-1]  # RGB->BGR
    return out

# ---------- CLI ----------
def cli_main():
    p = argparse.ArgumentParser(prog="deep.harmonize_cnn")
    p.add_argument("--in", dest="inp", required=True, help="panorama de entrada (BGR)")
    p.add_argument("--ref", help="imagem de referência (opcional)")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lam_tv", type=float, default=0.01)
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--out", default="pano_harmo.jpg")
    a = p.parse_args()

    pano = imread_color(a.inp)
    if a.ref:
        ref = imread_color(a.ref)
        pano, ref = resize_to_match(pano, ref)
        bbox = overlap_bbox(pano, ref)
        mask = None
        if bbox is not None:
            x0,y0,x1,y1 = bbox
            mask = np.zeros(pano.shape[:2], np.uint8)
            mask[y0:y1, x0:x1] = 1
        target = ref
    else:
        mask = None
        target = gray_world_target(pano)

    out = optimize_harmonizer(pano, target, mask, steps=a.steps, lr=a.lr, lam_tv=a.lam_tv, device=a.device)
    cv2.imwrite(a.out, out)
    print(f"ok: {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
