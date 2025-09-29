# modules/classic/metrics_plus.py
# Métricas avançadas: SSIM/MSE/PSNR + seam_energy + ghosting + DeltaE + overlap_ratio
# - ref + candidatos (--ref --imgs ...)
# - diretório (--dir) -> todas as combinações
# Opções: --align, --mask-nonzero, --min_side
import argparse, os, sys, csv, itertools, time
from typing import List, Dict
import numpy as np
import cv2

from modules.utils.metrics_core import (
    imread_color, resize_min_side, align_pair, valid_mask, compute_all
)

def apply_mask_pair(a: np.ndarray, b: np.ndarray, use_mask: bool) -> (np.ndarray, np.ndarray):
    if not use_mask:
        return a, b
    ma = valid_mask(a); mb = valid_mask(b)
    m = (ma & mb).astype(bool)
    if not np.any(m):
        return a, b
    # manter forma para SSIM: recortamos bbox da intersecção em vez de achatar
    ys, xs = np.where(m)
    y0, y1 = ys.min(), ys.max()+1
    x0, x1 = xs.min(), xs.max()+1
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1]

def evaluate_pair(path_a: str, path_b: str, args) -> dict:
    t0 = time.time()
    A = imread_color(path_a); B = imread_color(path_b)
    A = resize_min_side(A, args.min_side)
    B = resize_min_side(B, args.min_side)
    A, B = align_pair(A, B, args.align)
    A, B = apply_mask_pair(A, B, args.mask_nonzero)
    scores = compute_all(A, B)
    dt = time.time() - t0
    row = {
        "A": path_a, "B": path_b,
        "align": args.align,
        "mask_nonzero": int(args.mask_nonzero),
        "time_s": round(dt, 3),
    }
    row.update({k: (None if v is None else (float(v) if isinstance(v,(int,float,np.floating)) else v)) for k,v in scores.items()})
    return row

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
    if not rows: return
    cols = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_markdown(rows: List[dict], out_md: str, title: str):
    if not rows: return
    cols = list(rows[0].keys())
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")

def cli_main():
    p = argparse.ArgumentParser(prog="classic.metrics_plus")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ref", help="imagem de referência para comparar os demais")
    g.add_argument("--dir", help="diretório com imagens para avaliação pareada")
    p.add_argument("--imgs", nargs="+", help="candidatos quando usar --ref")
    p.add_argument("--align", choices=["resize","pad","crop"], default="resize")
    p.add_argument("--min_side", type=int, default=0)
    p.add_argument("--mask-nonzero", action="store_true")
    p.add_argument("--outdir", default="reports_metrics_plus")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = []

    if args.ref:
        if not args.imgs:
            sys.exit("use --imgs para fornecer os candidatos quando usar --ref")
        for c in args.imgs:
            rows.append(evaluate_pair(args.ref, c, args))
        title = "Métricas avançadas (referência vs candidatos)"
    else:
        paths = list_images_in_dir(args.dir)
        for a, b in itertools.combinations(paths, 2):
            rows.append(evaluate_pair(a, b, args))
        title = "Métricas avançadas (pares no diretório)"

    out_csv = os.path.join(args.outdir, "metrics_plus.csv")
    out_md  = os.path.join(args.outdir, "README.md")
    write_csv(rows, out_csv)
    write_markdown(rows, out_md, title)

    # resumo
    rows_sorted = sorted([r for r in rows if r.get("SSIM") is not None], key=lambda r: r["SSIM"], reverse=True)[:5]
    print(f"ok: {len(rows)} pares avaliados → {out_csv}")
    for r in rows_sorted:
        print(f"  • {os.path.basename(r['A'])} vs {os.path.basename(r['B'])} | SSIM={r['SSIM']:.4f} PSNR={r['PSNR']:.2f} dB | ghost={r['ghosting']:.3f} seamE={r['seam_energy']:.2f}")

    return 0

if __name__ == "__main__":
    sys.exit(cli_main())
