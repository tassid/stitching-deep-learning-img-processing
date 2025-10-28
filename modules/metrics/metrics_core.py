# modules/utils/metrics_core.py
# Núcleo de métricas para comparar panoramas e imagens alinhadas.
# Depende de: numpy, opencv-python, scikit-image (SSIM opcional)
from __future__ import annotations
import math, os
from typing import Dict, Tuple
import numpy as np
import cv2

try:
    from skimage.metrics import structural_similarity as ssim_sk
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# ---------- util de leitura/alinhamento ----------
def imread_color(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Falha ao ler: {path}")
    return im

def resize_min_side(img: np.ndarray, min_side: int) -> np.ndarray:
    if min_side <= 0: return img
    h, w = img.shape[:2]
    s = min(h, w)
    if s <= min_side: return img
    scale = float(min_side) / float(s)
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def align_pair(a: np.ndarray, b: np.ndarray, how: str) -> Tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) == (hb, wb):
        return a, b
    how = how.lower()
    if how == "resize":
        b2 = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_AREA)
        return a, b2
    if how == "pad":
        H, W = max(ha, hb), max(wa, wb)
        aa = np.zeros((H, W, 3), dtype=a.dtype); aa[:ha, :wa] = a
        bb = np.zeros((H, W, 3), dtype=b.dtype); bb[:hb, :wb] = b
        return aa, bb
    if how == "crop":
        H, W = min(ha, hb), min(wa, wb)
        return a[:H,:W], b[:H,:W]
    raise ValueError("align inválido: use resize|pad|crop")

def valid_mask(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray > 0).astype(np.uint8)

def overlap_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ma = valid_mask(a); mb = valid_mask(b)
    return (ma & mb).astype(np.uint8)

def overlap_ratio(a: np.ndarray, b: np.ndarray) -> float:
    m = overlap_mask(a,b)
    return float(m.sum()) / float(m.size)

def crop_to_overlap(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = overlap_mask(a,b)
    if not np.any(m):
        return a, b, m
    ys, xs = np.where(m>0)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1], m[y0:y1, x0:x1]

# ---------- métricas básicas ----------
def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None) -> float:
    """Calcula MSE, opcionalmente usando máscara para ignorar bordas pretas."""
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    d = a_f - b_f

    if mask is not None:
        # Aplicar máscara: considerar apenas pixels válidos
        mask_bool = mask.astype(bool)
        if mask_bool.ndim == 2:
            # Expandir máscara para 3 canais se necessário
            mask_bool = np.stack([mask_bool] * 3, axis=2)

        valid_pixels = mask_bool.sum()
        if valid_pixels == 0:
            return float('nan')

        return float(np.sum((d * mask_bool)**2) / valid_pixels)

    return float(np.mean(d*d))

def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None) -> float:
    """Calcula PSNR, opcionalmente usando máscara para ignorar bordas pretas."""
    m = mse(a, b, mask)
    if np.isnan(m) or m <= 1e-12:
        return float('inf')
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(m)

def ssim_gray(a: np.ndarray, b: np.ndarray, mask: np.ndarray = None) -> float:
    """Calcula SSIM em grayscale, opcionalmente usando máscara para ignorar bordas pretas."""
    ag = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    bg = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    # Se há máscara, cropar para região válida
    if mask is not None:
        mask_2d = mask if mask.ndim == 2 else mask[:,:,0]
        if np.any(mask_2d > 0):
            ys, xs = np.where(mask_2d > 0)
            y0, y1 = ys.min(), ys.max()+1
            x0, x1 = xs.min(), xs.max()+1
            ag = ag[y0:y1, x0:x1]
            bg = bg[y0:y1, x0:x1]
        else:
            return float('nan')

    if _HAS_SKIMAGE:
        s, _ = ssim_sk(ag, bg, full=True, data_range=255)
        return float(s)

    # Fallback simples
    C1 = (0.01*255)**2; C2 = (0.03*255)**2
    agf = ag.astype(np.float32); bgf = bg.astype(np.float32)
    mu1 = cv2.GaussianBlur(agf, (7,7), 1.5)
    mu2 = cv2.GaussianBlur(bgf, (7,7), 1.5)
    mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu12 = mu1*mu2
    sigma1_sq = cv2.GaussianBlur(agf*agf, (7,7), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(bgf*bgf, (7,7), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(agf*bgf, (7,7), 1.5) - mu12
    ssim_map = ((2*mu12 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + 1e-12)
    return float(np.mean(ssim_map))

# ---------- métricas fotométricas ----------
def mean_luminance(img: np.ndarray) -> float:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L = lab[...,0].astype(np.float32)  # 0..255
    return float(L.mean())

def deltaE_cie76_mean(a: np.ndarray, b: np.ndarray) -> float:
    # cv2 Lab é L in [0..255], a,b in [0..255] com offset; ainda funciona para DeltaE Euclidiano comparativo
    La = cv2.cvtColor(a, cv2.COLOR_BGR2Lab).astype(np.float32)
    Lb = cv2.cvtColor(b, cv2.COLOR_BGR2Lab).astype(np.float32)
    d = La - Lb
    de = np.sqrt(np.sum(d*d, axis=2))
    return float(de.mean())

def exposure_stats(a: np.ndarray, b: np.ndarray) -> dict:
    La = mean_luminance(a); Lb = mean_luminance(b)
    return {"deltaL_mean": float(La - Lb), "L_ref": La, "L_cmp": Lb}

# ---------- métricas de costura/ghosting ----------
def seam_energy(a: np.ndarray, b: np.ndarray) -> float:
    # energia média do gradiente da diferença |A-B| na zona de overlap
    ao, bo, m = crop_to_overlap(a,b)
    if not np.any(m):
        return float('nan')
    diff = cv2.absdiff(ao, bo)
    diff_g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(diff_g, cv2.CV_32F, 1,0,ksize=3)
    gy = cv2.Sobel(diff_g, cv2.CV_32F, 0,1,ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(mag.mean())

def ghosting_score(a: np.ndarray, b: np.ndarray, low: int = 100, high: int = 200) -> float:
    # 1 - IoU entre bordas (Canny) em overlap: mais alto = mais ghosting/misalignment
    ao, bo, m = crop_to_overlap(a,b)
    if not np.any(m):
        return float('nan')
    ea = cv2.Canny(cv2.cvtColor(ao, cv2.COLOR_BGR2GRAY), low, high)
    eb = cv2.Canny(cv2.cvtColor(bo, cv2.COLOR_BGR2GRAY), low, high)
    inter = np.logical_and(ea>0, eb>0).sum()
    union = np.logical_or (ea>0, eb>0).sum() + 1e-6
    iou = float(inter) / float(union)
    return 1.0 - iou  # maior = pior

# ---------- pacote completo ----------
def compute_all(a: np.ndarray, b: np.ndarray, use_mask: bool = True) -> dict:
    """
    Calcula todas as métricas entre duas imagens.

    Args:
        a: Imagem de referência (BGR)
        b: Imagem a ser comparada (BGR)
        use_mask: Se True, ignora bordas pretas (áreas com todos pixels = 0)

    Returns:
        Dict com todas as métricas calculadas
    """
    # Criar máscara de overlap se necessário
    mask = None
    if use_mask:
        mask = overlap_mask(a, b)
        # Verificar se há região válida
        if not np.any(mask):
            # Se não há overlap, retornar NaN para todas as métricas
            return {
                "SSIM": float('nan'),
                "MSE": float('nan'),
                "PSNR": float('nan'),
                "overlap_ratio": 0.0,
                "seam_energy": float('nan'),
                "ghosting": float('nan'),
                "deltaE_mean": float('nan'),
                "deltaL_mean": float('nan'),
                "L_ref": float('nan'),
                "L_cmp": float('nan'),
            }

    return {
        "SSIM": ssim_gray(a, b, mask),
        "MSE": mse(a, b, mask),
        "PSNR": psnr(a, b, mask),
        "overlap_ratio": overlap_ratio(a, b),
        "seam_energy": seam_energy(a, b),
        "ghosting": ghosting_score(a, b),
        "deltaE_mean": deltaE_cie76_mean(a, b),
        **exposure_stats(a, b),
    }
