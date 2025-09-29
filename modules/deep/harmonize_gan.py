# modules/deep/harmonize_gan.py
"""
Photometric Harmonization with GAN (Pix2Pix-style)
--------------------------------------------------
Treina rapidamente (ou usa pesos pré-treinados) um gerador (U-Net) condicionado
no panorama de entrada para harmonizar cor/exposição em relação a uma referência
ou alvo auto-supervisionado (gray-world + equalização leve).

- Gerador: U-Net (input 3ch, output 3ch)
- Discriminador: PatchGAN (70x70)
- Perdas: L1 (conteúdo), adversarial (BCE), e TV opcional no resíduo

Uso (treino rápido com referência):
  python -m modules.deep.harmonize_gan --in pano.jpg --ref uma_imagem.jpg --steps 1000 --lr 2e-4 --out pano_gan.jpg

Uso (inferência com pesos):
  python -m modules.deep.harmonize_gan --in pano.jpg --gen-weights g.pth --out pano_gan.jpg

Uso (sem referência; alvo auto-supervisionado):
  python -m modules.deep.harmonize_gan --in pano.jpg --steps 800 --lr 2e-4 --auto-target --out pano_gan.jpg

Requisitos:
  pip install opencv-contrib-python torch
"""

from __future__ import annotations
import argparse, sys, os
from typing import Optional, Tuple
import numpy as np
import cv2

# ---------- utils ----------
def imread_color(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    return im

def resize_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    tgt = img.astype(np.float32).copy()
    means = tgt.reshape(-1,3).mean(0) + 1e-6
    g = float(means.mean())
    gain = g / means
    tgt *= gain[None,None,:]
    tgt = np.clip(tgt, 0, 255)
    lab = cv2.cvtColor(tgt.astype(np.uint8), cv2.COLOR_BGR2Lab)
    lab[...,0] = cv2.equalizeHist(lab[...,0])
    tgt = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return tgt

# ---------- modelos ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, C=64):
        super().__init__()
        def down(cin, cout, bn=True):
            layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not bn)]
            if bn: layers += [nn.BatchNorm2d(cout)]
            layers += [nn.LeakyReLU(0.2, True)]
            return nn.Sequential(*layers)
        def up(cin, cout, dropout=False):
            layers = [nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(cout), nn.ReLU(True)]
            if dropout: layers += [nn.Dropout(0.5)]
            return nn.Sequential(*layers)

        self.d1 = down(in_ch, C, bn=False)   # 1/2
        self.d2 = down(C,   C*2)             # 1/4
        self.d3 = down(C*2, C*4)             # 1/8
        self.d4 = down(C*4, C*8)             # 1/16
        self.d5 = down(C*8, C*8)             # 1/32
        self.d6 = down(C*8, C*8)             # 1/64

        self.u6 = up(C*8, C*8, dropout=True)
        self.u5 = up(C*16, C*8, dropout=True)
        self.u4 = up(C*16, C*4)
        self.u3 = up(C*8,  C*2)
        self.u2 = up(C*4,  C)
        self.u1 = nn.Sequential(
            nn.ConvTranspose2d(C*2, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3); d5 = self.d5(d4); d6 = self.d6(d5)
        u6 = self.u6(d6)
        u5 = self.u5(torch.cat([u6, d5], dim=1))
        u4 = self.u4(torch.cat([u5, d4], dim=1))
        u3 = self.u3(torch.cat([u4, d3], dim=1))
        u2 = self.u2(torch.cat([u3, d2], dim=1))
        out = self.u1(torch.cat([u2, d1], dim=1))
        return out   # [-1,1]

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, C=64):
        super().__init__()
        def block(cin, cout, bn=True):
            layers = [nn.Conv2d(cin, cout, 4, 2, 1, bias=not bn)]
            if bn: layers += [nn.BatchNorm2d(cout)]
            layers += [nn.LeakyReLU(0.2, True)]
            return nn.Sequential(*layers)
        self.net = nn.Sequential(
            block(in_ch,   C, bn=False),
            block(C,     C*2),
            block(C*2,   C*4),
            nn.Conv2d(C*4, 1, 4, 1, 1)  # mapa de logits (PatchGAN)
        )
    def forward(self, x): return self.net(x)

# ---------- treino/inferência ----------
def to_tensor_bgr(img: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(img[:, :, ::-1].transpose(2,0,1)).float()[None]/255.0  # BGR->RGB
    return (t*2 - 1.0)  # [-1,1]

def to_image_bgr(t: torch.Tensor) -> np.ndarray:
    t = (t.clamp(-1,1)*0.5 + 0.5)  # [0,1]
    im = (t[0].permute(1,2,0).detach().cpu().numpy()*255.0).astype(np.uint8)
    return im[:, :, ::-1]  # RGB->BGR

def optimize_gan(pano: np.ndarray, target: np.ndarray, mask_overlap: Optional[np.ndarray],
                 steps=1000, lr=2e-4, lam_l1=100.0, device="cpu") -> np.ndarray:
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    x = to_tensor_bgr(pano).to(device)
    y = to_tensor_bgr(target).to(device)
    if mask_overlap is not None:
        m = torch.from_numpy((mask_overlap>0).astype(np.float32))[None,None].to(device)
    else:
        m = None

    for it in range(steps):
        # Discriminator
        with torch.no_grad():
            fake = G(x)
        D.zero_grad()
        pred_real = D(y)
        pred_fake = D(fake.detach())
        ones = torch.ones_like(pred_real)
        zeros = torch.zeros_like(pred_fake)
        loss_D = bce(pred_real, ones) + bce(pred_fake, zeros)
        loss_D.backward()
        optD.step()

        # Generator
        G.zero_grad()
        fake = G(x)
        pred_fake = D(fake)
        adv = bce(pred_fake, ones)
        if m is None:
            l1 = (fake - y).abs().mean()
        else:
            l1 = (((fake - y).abs()).mean(dim=1, keepdim=True)*m).sum() / (m.sum() + 1e-6)
        loss_G = adv + lam_l1*l1
        loss_G.backward()
        optG.step()

    with torch.no_grad():
        out = G(x)
    return to_image_bgr(out)

# ---------- CLI ----------
def cli_main():
    p = argparse.ArgumentParser(prog="deep.harmonize_gan")
    p.add_argument("--in", dest="inp", required=True, help="panorama de entrada (BGR)")
    p.add_argument("--ref", help="imagem de referência (opcional)")
    p.add_argument("--auto-target", action="store_true", help="usa alvo gray-world + equalização se ref ausente")
    p.add_argument("--gen-weights", help="arquivo .pth para inferência do gerador")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--out", default="pano_gan.jpg")
    a = p.parse_args()

    pano = imread_color(a.inp)

    if a.gen_weights:
        # inferência direta
        G = UNetGenerator().to(a.device)
        state = torch.load(a.gen_weights, map_location=a.device)
        G.load_state_dict(state, strict=False)
        G.eval()
        x = to_tensor_bgr(pano).to(a.device)
        with torch.no_grad():
            y = G(x)
        out = to_image_bgr(y)
        cv2.imwrite(a.out, out)
        print(f"ok (inferência): {a.out}")
        return 0

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
        if not a.auto_target:
            print("[aviso] sem --ref e sem --auto-target; usando alvo gray-world por padrão.")
        target = gray_world_target(pano)
        mask = None

    out = optimize_gan(pano, target, mask, steps=a.steps, lr=a.lr, device=a.device)
    cv2.imwrite(a.out, out)
    print(f"ok (treino rápido): {a.out}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
