# modules/classic/metrics.py
# Uso 1 (ref + candidatos):
#   python -m modules.classic.metrics --ref pano_ref.jpg --imgs pano_A.jpg pano_B.jpg --mask-nonzero --outdir reports_metrics
# Uso 2 (pasta com imagens):
#   python -m modules.classic.metrics --dir .\resultados --align resize --psnr --outdir reports_metrics
#
# Saídas: CSV (metrics.csv) e README.md (tabela) em --outdir

import argparse, os, sys, csv, itertools, math
from typing import List, Tuple, Optional
import numpy as np
import cv2

# ----- SSIM (via scikit-image, com fallback simples) -----
def ssim_skimage(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        s, _ = ssim(a_gray, b_gray, full=True, data_range=255)
        return float(s)
    except Exception:
        # Fallback: MSSIM aproximado (janela 7x7) – não tão robusto quanto skimage
        C1 = (0.01*255)**2
        C2 = (0.03*255)**2
        mu1 = cv2.GaussianBlur(a_gray.astype(np.float32), (7,7), 1.5)
        mu2 = cv2.GaussianBlur(b_gray.astype(np.float32), (7,7), 1.5)
        mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu12 = mu1*mu2
        sigma1_sq = cv2.GaussianBlur(a_gray.astype(np.float32)*a_gray.astype(np.float32), (7,7), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(b_gray.astype(np.float32)*b_gray.astype(np.float32), (7,7), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(a_gray.astype(np.float32)*b_gray.astype(np.float32), (7,7), 1.5) - mu12
        ssim_map = ((2*mu12 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2) + 1e-12)
        return float(np.mean(ssim_map))

def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff*diff))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return float('inf')
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(m)

# ----- leitura/alinhamento -----
def imread_color(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        sys.exit(f"Falha ao ler: {path}")
    return im

def resize_min_side(img: np.ndarray, min_side: int) -> np.ndarray:
    if min_side <= 0:
        return img
    h, w = img.shape[:2]
    s = min(h, w)
    if s <= min_side:
        return img
    scale = min_side / s
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

def align_pair(a: np.ndarray, b: np.ndarray, how: str) -> Tuple[np.ndarray, np.ndarray]:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    if (ha, wa) == (hb, wb):
        return a, b
    how = how.lower()
    if how == "resize":
        # redimensiona B para A
        b2 = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_AREA)
        return a, b2
    elif how == "pad":
        H, W = max(ha, hb), max(wa, wb)
        aa = np.zeros((H, W, 3), dtype=a.dtype); aa[:ha, :wa] = a
        bb = np.zeros((H, W, 3), dtype=b.dtype); bb[:hb, :wb] = b
        return aa, bb
    elif how == "crop":
        H, W = min(ha, hb), min(wa, wb)
        return a[:H,:W], b[:H,:W]
    else:
        sys.exit("align inválido: use resize|pad|crop")

def make_valid_mask(img: np.ndarray) -> np.ndarray:
    # pixels não pretos contam como válidos (útil para mosaicos com fundo 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return (gray > 0).astype(np.uint8)

def apply_mask(a: np.ndarray, b: np.ndarray, mask_nonzero: bool) -> Tuple[np.ndarray, np.ndarray]:
    if not mask_nonzero:
        return a, b
    ma = make_valid_mask(a)
    mb = make_valid_mask(b)
    m = (ma & mb).astype(bool)
    if not np.any(m):
        # sem interseção – volta sem máscara para não quebrar
        return a, b
    a2 = a.copy(); b2 = b.copy()
    a2 = a2[m]; b2 = b2[m]
    # achatar em pixels válidos (N,3) → (H',W',3) fictício não necessário; SSIM usa gray com reshape
    return a2, b2

def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----- núcleo de avaliação -----
def evaluate_pair(path_a: str, path_b: str, args) -> dict:
    a = imread_color(path_a); b = imread_color(path_b)
    a = resize_min_side(a, args.min_side)
    b = resize_min_side(b, args.min_side)
    a, b = align_pair(a, b, args.align)

    # máscara opcional da área válida
    if args.mask_nonzero:
        # Para SSIM precisamos manter forma 2D; então pegamos apenas os pixels válidos e
        # reconstruímos "linhas" virtuais.
        # Estratégia: mascara na versão em cinza e achata.
        ag = to_gray(a); bg = to_gray(b)
        ma = (ag > 0); mb = (bg > 0); m = ma & mb
        if np.any(m):
            ag = ag[m].reshape(-1, 1)
            bg = bg[m].reshape(-1, 1)
            # Para MSE/PSNR em cor, aplicamos a mesma máscara nas 3 bandas
            a_c = a[m]; b_c = b[m]
        else:
            ag = to_gray(a); bg = to_gray(b)
            a_c = a; b_c = b
    else:
        ag = to_gray(a); bg = to_gray(b)
        a_c = a; b_c = b

    # SSIM em escala de cinza
    # (se usamos os vetores achatados, adaptamos para 2D “fake” via reshape longo)
    if ag.ndim == 2:
        ssim_val = ssim_skimage(ag, bg)
    else:
        # reshape em “imagem” 2D longa
        n = ag.shape[0]
        w = int(np.sqrt(n))
        h = n // w
        ag2 = ag[:h*w].reshape(h, w).astype(np.uint8)
        bg2 = bg[:h*w].reshape(h, w).astype(np.uint8)
        ssim_val = ssim_skimage(ag2, bg2)

    mse_val = mse(a_c, b_c)
    row = {
        "A": path_a,
        "B": path_b,
        "align": args.align,
        "mask_nonzero": int(args.mask_nonzero),
        "SSIM": round(ssim_val, 6),
        "MSE": round(mse_val, 6),
    }
    if args.psnr:
        row["PSNR"] = round(psnr(a_c, b_c), 6)
    return row

# ----- utilidades de IO -----
def list_images_in_dir(d: str) -> List[str]:
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    paths = []
    for fname in sorted(os.listdir(d)):
        if os.path.splitext(fname.lower())[1] in exts:
            paths.append(os.path.join(d, fname))
    if not paths:
        sys.exit(f"Sem imagens em: {d}")
    return paths

def write_csv(rows: List[dict], out_csv: str):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_markdown(rows: List[dict], out_md: str, title: str):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")

# ----- CLI -----
def cli_main():
    p = argparse.ArgumentParser(prog="classic.metrics")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref", help="imagem de referência para comparar os demais")
    g.add_argument("--dir", help="diretório com imagens para avaliação pareada")
    p.add_argument("--imgs", nargs="+", help="candidatos quando usar --ref")
    p.add_argument("--align", choices=["resize","pad","crop"], default="resize", help="alinhamento de tamanho")
    p.add_argument("--min_side", type=int, default=0, help="redimensiona lado menor (0 = não)")
    p.add_argument("--mask-nonzero", action="store_true", help="usa só área válida (não-preta) em ambos")
    p.add_argument("--psnr", action="store_true", help="inclui PSNR")
    p.add_argument("--outdir", default="reports_metrics")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = []

    if args.ref:
        if not args.imgs:
            sys.exit("use --imgs para fornecer os candidatos quando usar --ref")
        ref_path = args.ref
        for c in args.imgs:
            rows.append(evaluate_pair(ref_path, c, args))
        title = "Métricas (referência vs candidatos)"
    else:
        paths = list_images_in_dir(args.dir)
        for a, b in itertools.combinations(paths, 2):
            rows.append(evaluate_pair(a, b, args))
        title = "Métricas pareadas no diretório"

    out_csv = os.path.join(args.outdir, "metrics.csv")
    out_md  = os.path.join(args.outdir, "README.md")
    write_csv(rows, out_csv)
    write_markdown(rows, out_md, title)

    # resumo no console (top-5 por SSIM)
    rows_sorted = sorted(rows, key=lambda r: r["SSIM"], reverse=True)[:5]
    print(f"ok: {len(rows)} pares avaliados → {out_csv}")
    for r in rows_sorted:
        pair = f"{os.path.basename(r['A'])} vs {os.path.basename(r['B'])}"
        ssim_str = f"SSIM={r['SSIM']}"
        mse_str = f"MSE={r['MSE']}"
        ps = f", PSNR={r['PSNR']}" if 'PSNR' in r else ""
        print(f"  • {pair}: {ssim_str}, {mse_str}{ps}")
    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
