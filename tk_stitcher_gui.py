# -*- coding: utf-8 -*-
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import json
import time
from PIL import Image, ImageTk

# ===== Dependências =====
try:
    import cv2
except Exception as err:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("Erro", "OpenCV não encontrado. Instale com: pip install opencv-python\nDetalhes: " + str(err))
    raise

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ===== Tooltip Helper =====
class ToolTip:
    """Cria tooltips para widgets"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip, text=self.text, background="#ffffe0",
                        relief=tk.SOLID, borderwidth=1, font=("Arial", 9))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# OpenCV constants (compat)
ST_OK = getattr(cv2, "STITCHER_OK", 0)
ST_ERR_NEED_MORE_IMGS = getattr(cv2, "STITCHER_ERR_NEED_MORE_IMGS", 1)
ST_ERR_HOMOGRAPHY_EST_FAIL = getattr(cv2, "STITCHER_ERR_HOMOGRAPHY_EST_FAIL", 2)
ST_ERR_CAMERA_PARAMS_ADJUST_FAIL = getattr(cv2, "STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL", 3)

def status_to_msg(status: int) -> str:
    mapping = {
        ST_ERR_NEED_MORE_IMGS: "Não há imagens suficientes (>=2) ou sobreposição insuficiente.",
        ST_ERR_HOMOGRAPHY_EST_FAIL: "Falha ao estimar homografia (poucas features ou não correspondem).",
        ST_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Falha ao ajustar parâmetros da câmera.",
    }
    tip = " Dicas: mantenha ordem/seq, 30-50% de overlap, iluminação semelhante; use modo SCANS para varreduras."
    return mapping.get(status, "Falhou com status " + str(status) + ".") + tip

def make_stitcher(modo: str, try_gpu: bool):
    mode = cv2.Stitcher_PANORAMA if modo == "panorama" else cv2.Stitcher_SCANS
    st = cv2.Stitcher_create(mode)
    if hasattr(st, "setPanoConfidenceThresh"):
        st.setPanoConfidenceThresh(0.5)
    try:
        Warper = getattr(cv2, "PyRotationWarper", None)
        if Warper and hasattr(st, "setWarper"):
            st.setWarper(Warper("spherical", 1.0))
    except Exception:
        pass
    _ = try_gpu  # Stitcher de pip não usa CUDA
    return st

def read_images(files: List[Path]):
    imgs = []
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError("Falha ao ler: " + str(p))
        imgs.append(im)
    if len(imgs) < 2:
        raise RuntimeError("Selecione pelo menos 2 imagens.")
    return imgs

# ===== Utils de MÉTRICAS =====
def save_metrics_json(path: Path, metrics: dict):
    Path(path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

def save_metrics_txt(path: Path, method_key: str, ts: str, metrics: dict, out_name: str, ref_path: str):
    lines = [
        f"timestamp: {ts}",
        f"method: {method_key}",
        f"output: {out_name}",
        f"reference: {ref_path}",
        f"psnr_db: {metrics.get('psnr', float('nan')):.3f}",
        f"ssim: {metrics.get('ssim', float('nan')):.4f}",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

def append_metrics_log(output_dir: Path, key: str, ts: str, metrics: dict, extra: dict | None = None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    line = f"{ts} [{key}] PSNR={metrics.get('psnr', float('nan')):.3f} SSIM={metrics.get('ssim', float('nan')):.4f}"
    if extra:
        line += " " + " ".join(f"{k}={v}" for k, v in extra.items())
    line += "\n"
    (output_dir / "metrics_log.txt").open("a", encoding="utf-8").write(line)

def create_before_after_montage(input_imgs: List[Path], ref_img_path: Path, result_img_path: Path, output_path: Path, max_height: int = 400):
    """
    Cria montagem comparativa com:
    - Linha 1: imagens originais (01, 02, 03...)
    - Linha 2: imagem de referência + imagem resultante

    Args:
        input_imgs: Lista de caminhos das imagens originais
        ref_img_path: Caminho da imagem de referência do dataset
        result_img_path: Caminho da imagem resultante do método
        output_path: Onde salvar a montagem
        max_height: Altura máxima para redimensionar (mantém proporção)
    """
    import numpy as np

    def resize_keep_aspect(img, max_h):
        h, w = img.shape[:2]
        if h > max_h:
            scale = max_h / h
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
        return img

    # Carregar e redimensionar imagens originais
    originals = []
    for p in input_imgs:
        img = cv2.imread(str(p))
        if img is not None:
            img = resize_keep_aspect(img, max_height)
            originals.append(img)

    if not originals:
        raise RuntimeError("Nenhuma imagem original válida para montagem")

    # Carregar referência e resultado
    ref_img = cv2.imread(str(ref_img_path))
    result_img = cv2.imread(str(result_img_path))

    if ref_img is None:
        raise RuntimeError(f"Não foi possível carregar imagem de referência: {ref_img_path}")
    if result_img is None:
        raise RuntimeError(f"Não foi possível carregar imagem resultante: {result_img_path}")

    # Redimensionar referência e resultado
    ref_img = resize_keep_aspect(ref_img, max_height)
    result_img = resize_keep_aspect(result_img, max_height)

    # Adicionar labels nas imagens
    def add_label(img, text, color=(255, 255, 255)):
        h, w = img.shape[:2]
        img_copy = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        # Fundo preto semi-transparente para o texto
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (10, h - text_h - 20), (20 + text_w, h - 10), (0, 0, 0), -1)
        img_copy = cv2.addWeighted(overlay, 0.6, img_copy, 0.4, 0)
        cv2.putText(img_copy, text, (15, h - 15), font, font_scale, color, thickness, cv2.LINE_AA)
        return img_copy

    # Adicionar labels
    for i, img in enumerate(originals):
        originals[i] = add_label(img, f"Input {i+1:02d}")
    ref_img = add_label(ref_img, "Reference")
    result_img = add_label(result_img, "Result")

    # Criar linha 1: imagens originais
    row1 = np.hstack(originals)

    # Criar linha 2: referência + resultado
    row2_imgs = [ref_img, result_img]
    row2 = np.hstack(row2_imgs)

    # Ajustar larguras para que as linhas tenham o mesmo tamanho
    max_width = max(row1.shape[1], row2.shape[1])

    # Preencher com branco se necessário
    if row1.shape[1] < max_width:
        padding = np.ones((row1.shape[0], max_width - row1.shape[1], 3), dtype=np.uint8) * 255
        row1 = np.hstack([row1, padding])

    if row2.shape[1] < max_width:
        padding = np.ones((row2.shape[0], max_width - row2.shape[1], 3), dtype=np.uint8) * 255
        row2 = np.hstack([row2, padding])

    # Combinar linhas
    montage = np.vstack([row1, row2])

    # Salvar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), montage)

    return montage

# ===== Registry de métodos =====
METHODS = {}

# Descrições dos métodos para tooltips
METHOD_DESCRIPTIONS = {
    "classic/1_stitcher_basico": "SIFT + homografia simples. Base sólida e rápida.",
    "classic/2_manual_sift_ransac_feather": "SIFT manual com feather blending suave.",
    "classic/3_features_homography": "Features clássicos (SIFT/ORB) com múltiplas opções.",
    "classic/4_blend_exposure": "Feather blend suave baseado em distância. Natural e rápido.",
    "classic/5_blend_wavelet": "Fusão por wavelets (Daubechies). Preserva detalhes finos.",
    "classic/6_seams_exposure_multiband": "Graph-cut seams + multiband blending. Resultado profissional.",
    "classic/7_detectors_harris_fast": "Detectores de cantos Harris/FAST. Leve e rápido.",
    "classic/8_blend_laplacian": "Pirâmide Laplaciana para blend multi-escala. Suave e natural.",
    "classic/9_exposure_comp_only": "Compensação de exposição + feather. Remove diferenças de iluminação.",
    "classic/10_blend_poisson": "Seamless cloning de Poisson nas áreas de overlap. Ótimo para remover costuras.",
    "classic/11_blend_multifocal": "Blend baseado em foco. Combina áreas nítidas de múltiplas imagens.",
    "classic/12_blend_hybrid": "Combina contraste, saturação e exposição. Flexível e personalizável.",
    "classic/13_ecc_alignment": "Enhanced Correlation Coefficient para alinhamento preciso.",
    "classic/opencv_stitcher": "Stitcher oficial do OpenCV com auto-detecção. Rápido e confiável.",
    "classic/orb_homography": "Features ORB + RANSAC. Bom para imagens com pouca textura.",
    "deep/deep_homography": "CNN para estimativa de homografia. Robusto a grandes deslocamentos.",
    "deep/flow_stitch": "SIFT robusto com fallback. Confiável para panoramas gerais.",
    "deep/loftr": "Detector-free matching. Excelente para pares com pouca textura.",
    "deep/sp_superglue": "SIFT com blend adaptativo. Robusto e versátil.",
    "deep/udis": "SIFT simplificado. Rápido e eficiente.",
}

def register_method(key, label, description=""):
    def deco(fn):
        METHODS[key] = {"label": label, "fn": fn, "accepts_try_gpu": True, "description": description}
        return fn
    return deco

# Descoberta automática de algorithms em modules/classic e modules/deep
def discover_algorithms():
    import importlib, pkgutil, sys, tempfile, shutil, subprocess, inspect
    from pathlib import Path
    import cv2 as _cv

    def add_bridge(mod_name: str, key: str, label: str):
        # Caminho 1: import direto com stitch(imgs[, try_gpu|device])
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, "stitch", None)
            if callable(fn):
                sig = inspect.signature(fn)
                accepted = set(sig.parameters.keys())
                def _call(imgs, **kwargs):
                    kw = {}
                    if "try_gpu" in accepted and "try_gpu" in kwargs:
                        kw["try_gpu"] = kwargs["try_gpu"]
                    if "device" in accepted and "device" in kwargs:
                        kw["device"] = kwargs["device"]
                    return fn(imgs, **kw) if kw else fn(imgs)
                METHODS[key] = {
                    "label": label,
                    "fn": _call,
                    "accepts_try_gpu": ("try_gpu" in accepted or "device" in accepted),
                }
                return
        except Exception:
            pass
        # Caminho 2: subprocesso com --dir/--out e opcional --device
        def _bridge(imgs, **kwargs):
            tmp = tempfile.mkdtemp(prefix=key.replace("/", "_") + "_")
            try:
                in_dir = Path(tmp) / "imgs"
                in_dir.mkdir(parents=True, exist_ok=True)
                for i, im in enumerate(imgs):
                    _cv.imwrite(str(in_dir / f"im_{i:03d}.jpg"), im)
                out_file = Path(tmp) / "out.jpg"
                cmd = [sys.executable, "-m", mod_name, "--dir", str(in_dir), "--out", str(out_file)]
                if kwargs.get("try_gpu", False):
                    try:
                        import torch  # type: ignore
                        if torch.cuda.is_available():
                            cmd += ["--device", "cuda"]
                        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                            cmd += ["--device", "mps"]
                    except Exception:
                        pass
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0:
                    raise RuntimeError(f"{label} falhou (retcode {ret.returncode})\nSTDOUT:\n{ret.stdout}\nSTDERR:\n{ret.stderr}")
                pano = _cv.imread(str(out_file))
                if pano is None:
                    raise RuntimeError(label + " não gerou imagem em: " + str(out_file))
                return pano
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        METHODS[key] = {"label": label, "fn": _bridge, "accepts_try_gpu": True}

    # Clássicos
    try:
        import modules.classic as classic_pkg
        for info in pkgutil.iter_modules(classic_pkg.__path__):
            name = info.name
            if name.startswith("__"):
                continue
            # Gerar label mais limpo removendo números do início
            clean_name = name
            if name[0].isdigit():
                # Remove "1_" ou "10_" etc do início para o label
                parts = name.split("_", 1)
                clean_name = parts[1] if len(parts) > 1 else name
            add_bridge(f"modules.classic.{name}", f"classic/{name}", "Clássico • " + clean_name.replace("_", " ").title())
    except Exception:
        pass

    # Deep
    try:
        import modules.deep as deep_pkg
        # Lista de módulos que NÃO são para stitching (harmonização/pós-processamento)
        skip_modules = {"harmonize_cnn", "harmonize_gan"}
        for info in pkgutil.iter_modules(deep_pkg.__path__):
            name = info.name
            if name.startswith("__") or name in skip_modules:
                continue
            add_bridge(f"modules.deep.{name}", f"deep/{name}", "Deep • " + name.replace("_", " ").title())
    except Exception:
        pass

    # Garantir presença dos dois métodos nativos
    METHODS.setdefault("classic/opencv_stitcher", {"label": "Clássico • OpenCV Stitcher (panorama/scans)", "fn": method_opencv_stitcher, "accepts_try_gpu": False})
    METHODS.setdefault("classic/orb_homography", {"label": "Clássico • ORB + Homography (sequencial)", "fn": method_orb_homography, "accepts_try_gpu": False})

@register_method("classic/opencv_stitcher", "Clássico • OpenCV Stitcher (panorama/scans)")
def method_opencv_stitcher(imgs, modo="panorama", try_gpu=False):
    st = make_stitcher(modo, try_gpu)
    status, pano = st.stitch(imgs)
    if status != ST_OK:
        raise RuntimeError(status_to_msg(status))
    return pano

@register_method("classic/orb_homography", "Clássico • ORB + Homography (sequencial)")
def method_orb_homography(imgs, ratio=0.75, ransac_reproj=4.0, try_gpu: bool=False):
    import numpy as np
    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match12(a, b):
        k1, d1 = orb.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), None)
        k2, d2 = orb.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
        if d1 is None or d2 is None:
            raise RuntimeError("Descritores ORB insuficientes.")
        knn = bf.knnMatch(d2, d1, k=2)  # b -> a
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m0, m1 = pair
            if m0.distance < ratio * m1.distance:
                good.append(m0)
        if len(good) < 8:
            raise RuntimeError("Correspondências insuficientes para homografia.")
        pts_b = np.float32([k2[m.queryIdx].pt for m in good])
        pts_a = np.float32([k1[m.trainIdx].pt for m in good])
        H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, ransac_reproj)
        if H is None:
            raise RuntimeError("Falha ao estimar homografia (RANSAC).")
        return H

    def warp_merge(base, img, H):
        h1, w1 = base.shape[:2]
        h2, w2 = img.shape[:2]
        corners_img = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        corners_img_w = cv2.perspectiveTransform(corners_img, H)
        corners_base = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        all_pts = np.vstack((corners_img_w, corners_base))
        xmin, ymin = np.floor(all_pts.min(axis=0).ravel()).astype(int)
        xmax, ymax = np.ceil(all_pts.max(axis=0).ravel()).astype(int)
        tx, ty = -xmin, -ymin
        T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=float)
        size = (xmax - xmin, ymax - ymin)
        base_w = cv2.warpPerspective(base, T, size)
        img_w = cv2.warpPerspective(img, T @ H, size)
        mask_b = (base_w.sum(axis=2) > 0).astype("uint8") * 255
        mask_i = (img_w.sum(axis=2) > 0).astype("uint8") * 255
        overlap = cv2.bitwise_and(mask_b, mask_i)
        out = base_w.copy()
        out[mask_i > 0] = img_w[mask_i > 0]
        if overlap.any():
            mix = (base_w.astype(float) + img_w.astype(float)) / 2.0
            out[overlap > 0] = mix[overlap > 0].astype("uint8")
        return out

    pano = imgs[0]
    for i in range(1, len(imgs)):
        H = match12(pano, imgs[i])
        pano = warp_merge(pano, imgs[i], H)
    return pano

@register_method("deep/loftr", "Deep • LoFTR (via subprocess)")
def method_loftr_bridge(imgs, **kwargs):
    """Bridge para modules/deep/loftr.py: aceita --dir/--out e opcional --device (cuda/mps)."""
    import tempfile, shutil, subprocess, sys
    from pathlib import Path
    import cv2 as _cv
    tmp = tempfile.mkdtemp(prefix="loftr_in_")
    try:
        in_dir = Path(tmp) / "imgs"
        in_dir.mkdir(parents=True, exist_ok=True)
        for i, im in enumerate(imgs):
            _cv.imwrite(str(in_dir / f"im_{i:03d}.jpg"), im)
        out_file = Path(tmp) / "out.jpg"
        cmd = [sys.executable, "-m", "modules.deep.loftr", "--dir", str(in_dir), "--out", str(out_file)]
        if kwargs.get("try_gpu", False):
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    cmd += ["--device", "cuda"]
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    cmd += ["--device", "mps"]
            except Exception:
                pass
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            raise RuntimeError(f"LoFTR falhou (retcode {ret.returncode})\nSTDOUT:\n{ret.stdout}\nSTDERR:\n{ret.stderr}")
        pano = _cv.imread(str(out_file))
        if pano is None:
            raise RuntimeError("LoFTR concluiu mas não gerou imagem em: " + str(out_file))
        return pano
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

# SuperGlue e Flow Stitch agora são descobertos automaticamente via discover_algorithms()
# Os placeholders abaixo foram removidos pois os módulos reais estão em:
# - modules/deep/sp_superglue.py (chave: deep/sp_superglue)
# - modules/deep/flow_stitch.py (chave: deep/flow_stitch)

class StitcherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stitcher — Clássico & Deep")
        self.geometry("1020x640")
        self.minsize(940, 600)
        self.configure(padx=10, pady=10)

        discover_algorithms()

        self.files: List[Path] = []
        # >>> resultados sempre na pasta do arquivo
        self.output_dir = Path(__file__).resolve().parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.modo = tk.StringVar(value="panorama")
        self.try_gpu = tk.BooleanVar(value=False)

        # Criar mapeamento label -> key para o dropdown
        self.label_to_key = {METHODS[k]["label"]: k for k in METHODS}
        self.key_to_label = {k: METHODS[k]["label"] for k in METHODS}

        # Usar label no dropdown, mas armazenar key internamente
        first_label = self.key_to_label[list(METHODS.keys())[0]]
        self.method_label = tk.StringVar(value=first_label)

        self.do_metrics = tk.BooleanVar(value=False)
        self.ref_path = tk.StringVar(value="")
        self.do_before_after = tk.BooleanVar(value=False)  # Nova opção para montagens before/after
        # >>> crie a StringVar ANTES de construir a UI da direita
        self.out_dir_var = tk.StringVar(value=str(self.output_dir))

        # Novas variáveis para melhorias
        self.last_result_path: Optional[Path] = None
        self.execution_start_time: Optional[float] = None
        self.thumbnails = {}  # Cache de thumbnails
        self.progress_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar(value="Pronto")
        self.time_var = tk.StringVar(value="--:--")
        self.presets = {}  # Presets salvos
        self.history = []  # Histórico de execuções

        self._build_left()
        self._build_right()
        self._build_status_bar()
        self._setup_shortcuts()
        self._setup_drag_drop()
        self._update_status()

        # Carregar presets e histórico
        self._load_presets()
        self._load_history()

    # ---------- UI ----------
    def _build_left(self):
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bar = ttk.Frame(left)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Adicionar imagens...", command=self.add_files).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Adicionar pasta...", command=self.add_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Limpar lista", command=self.clear_files).pack(side=tk.LEFT)

        self.listbox = tk.Listbox(left, selectmode=tk.EXTENDED)
        yscroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=yscroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(8, 0))
        yscroll.pack(side=tk.LEFT, fill=tk.Y, pady=(8, 0))

        # Contador de imagens
        self.img_count_label = ttk.Label(left, text="0 imagens", font=("", 9, "italic"))
        self.img_count_label.pack(anchor=tk.W, pady=(4, 0))

        sort_bar = ttk.Frame(left)
        sort_bar.pack(fill=tk.X, pady=6)
        ttk.Button(sort_bar, text="▲ Subir", command=lambda: self.move_selection(-1)).pack(side=tk.LEFT)
        ttk.Button(sort_bar, text="▼ Descer", command=lambda: self.move_selection(1)).pack(side=tk.LEFT, padx=6)
        ttk.Button(sort_bar, text="Remover selecionados", command=self.remove_selected).pack(side=tk.LEFT)

        # Preview da primeira imagem
        preview_frame = ttk.LabelFrame(left, text="Preview")
        preview_frame.pack(fill=tk.X, pady=6)
        self.preview_label = ttk.Label(preview_frame, text="Selecione imagens...", anchor=tk.CENTER)
        self.preview_label.pack(pady=10, fill=tk.BOTH, expand=True)

    def _build_right(self):
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(10, 0))

        box = ttk.LabelFrame(right, text="Parâmetros")
        box.pack(fill=tk.X)

        meth_row = ttk.Frame(box)
        meth_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(meth_row, text="Método:").pack(side=tk.LEFT)
        # Ordenar: classics primeiro, depois deep (usando labels, não keys)
        all_keys = list(METHODS.keys())
        classic_keys = sorted([k for k in all_keys if k.startswith("classic/")])
        deep_keys = sorted([k for k in all_keys if k.startswith("deep/")])
        ordered_keys = classic_keys + deep_keys
        # Converter keys para labels para exibição
        values = [self.key_to_label[k] for k in ordered_keys]
        self.method_cb = ttk.Combobox(meth_row, textvariable=self.method_label, state="readonly", width=60, values=values)
        self.method_cb.pack(side=tk.LEFT, padx=8)
        self.method_cb.bind("<<ComboboxSelected>>", self._on_method_change)

        mode_row = ttk.Frame(box)
        mode_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mode_row, text="Modo:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Panorama", value="panorama", variable=self.modo).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(mode_row, text="Scans", value="scans", variable=self.modo).pack(side=tk.LEFT, padx=(8, 0))

        gpu_row = ttk.Frame(box)
        gpu_row.pack(fill=tk.X, pady=6)
        ttk.Checkbutton(gpu_row, text="Tentar GPU (Deep/LoFTR)", variable=self.try_gpu).pack(side=tk.LEFT)

        # Presets
        preset_box = ttk.LabelFrame(right, text="Presets")
        preset_box.pack(fill=tk.X, pady=(10, 0))
        preset_row1 = ttk.Frame(preset_box)
        preset_row1.pack(fill=tk.X, pady=6)
        ttk.Button(preset_row1, text="💾 Salvar Preset", command=self.save_preset).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(preset_row1, text="📂 Carregar Preset", command=self.load_preset).pack(side=tk.LEFT)

        met_box = ttk.LabelFrame(right, text="Métricas (opcional)")
        met_box.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(met_box, text="Calcular métricas vs referência (PSNR/SSIM)", variable=self.do_metrics).pack(anchor=tk.W)
        ref_row = ttk.Frame(met_box)
        ref_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(ref_row, text="Referência:").pack(side=tk.LEFT)
        ttk.Entry(ref_row, textvariable=self.ref_path, width=44).pack(side=tk.LEFT, padx=6)
        ttk.Button(ref_row, text="Escolher...", command=self.choose_ref).pack(side=tk.LEFT)
        # Checkbox para montagens before/after
        ttk.Checkbutton(met_box, text="Gerar montagens comparativas (Before/After)", variable=self.do_before_after).pack(anchor=tk.W, pady=(6, 0))

        out_box = ttk.LabelFrame(right, text="Saída")
        out_box.pack(fill=tk.X, pady=(10, 0))
        out_row = ttk.Frame(out_box)
        out_row.pack(fill=tk.X, pady=6)
        ttk.Label(out_row, text="Pasta:").pack(side=tk.LEFT)
        self.out_entry = ttk.Entry(out_row, textvariable=self.out_dir_var, width=44)
        self.out_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(out_row, text="Escolher...", command=self.choose_output_dir).pack(side=tk.LEFT)

        act_box = ttk.Frame(right)
        act_box.pack(fill=tk.X, pady=(12, 0))
        self.run_btn = ttk.Button(act_box, text="▶ Executar", command=self.run_stitch)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.run_all_btn = ttk.Button(act_box, text="▶▶ Todos", command=self.run_all_methods)
        self.run_all_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.scan_folder_btn = ttk.Button(act_box, text="📁 Varrer Pasta", command=self.scan_folder)
        self.scan_folder_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.rerun_btn = ttk.Button(act_box, text="🔄 Re-executar", command=self.rerun_last, state=tk.DISABLED)
        self.rerun_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.open_result_btn = ttk.Button(act_box, text="🖼 Abrir", command=self.open_result, state=tk.DISABLED)
        self.open_result_btn.pack(side=tk.LEFT)

        # Botões de utilidades
        util_box = ttk.Frame(right)
        util_box.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(util_box, text="📊 Histórico", command=self.show_history, width=12).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(util_box, text="📄 Relatório", command=self.export_batch_report, width=12).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(util_box, text="❓ Ajuda", command=self.show_help, width=12).pack(side=tk.LEFT)

        # Status e progresso em nova linha
        status_box = ttk.Frame(right)
        status_box.pack(fill=tk.X, pady=(6, 0))
        self.progress = ttk.Progressbar(status_box, mode="determinate", maximum=100, length=300)
        self.progress.pack(side=tk.LEFT, padx=(0, 8))
        self.progress_label = ttk.Label(status_box, text="0%", width=4)
        self.progress_label.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(status_box, textvariable=self.time_var, width=8).pack(side=tk.LEFT)

        log_box = ttk.LabelFrame(right, text="Log")
        log_box.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log = tk.Text(log_box, height=18, width=48, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

    # ---------- Handlers ----------
    def add_files(self):
        paths = filedialog.askopenfilenames(title="Selecione imagens",
                                            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")])
        if not paths:
            return
        new_files = [Path(p) for p in paths if Path(p).suffix.lower() in SUPPORTED_EXTS]
        self._append_files(new_files)

    def add_folder(self):
        folder = filedialog.askdirectory(title="Escolha a pasta com imagens")
        if not folder:
            return
        folder = Path(folder)
        new_files = sorted([p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS])
        if not new_files:
            messagebox.showwarning("Aviso", "Nenhuma imagem suportada encontrada na pasta.")
            return
        self._append_files(new_files)

    def _append_files(self, new_files: List[Path]):
        existing = set(map(str, self.files))
        add = 0
        for p in new_files:
            if str(p) not in existing:
                self.files.append(p)
                self.listbox.insert(tk.END, str(p))
                add += 1
        self._log(f"Adicionados {add} arquivo(s). Total: {len(self.files)}")
        self._update_status()
        self._show_preview()

    def clear_files(self):
        self.files.clear()
        self.listbox.delete(0, tk.END)
        self._log("Lista limpa.")
        self._update_status()
        self._show_preview()

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            self.listbox.delete(idx)
            del self.files[idx]
        self._log(f"Removidos {len(sel)} item(ns).")
        self._update_status()
        self._show_preview()

    def move_selection(self, delta: int):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for i in (sel if delta > 0 else reversed(sel)):
            j = i + delta
            if 0 <= j < self.listbox.size():
                self.files[i], self.files[j] = self.files[j], self.files[i]
                txt_i, txt_j = self.listbox.get(i), self.listbox.get(j)
                self.listbox.delete(i); self.listbox.insert(i, txt_j)
                self.listbox.delete(j); self.listbox.insert(j, txt_i)
        self.listbox.selection_clear(0, tk.END)
        for i in [k + delta for k in sel if 0 <= k + delta < self.listbox.size()]:
            self.listbox.selection_set(i)

    def choose_output_dir(self):
        d = filedialog.askdirectory(title="Escolha a pasta de saída", initialdir=str(self.output_dir))
        if d:
            self.output_dir = Path(d)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.out_dir_var.set(str(self.output_dir))

    def choose_ref(self):
        f = filedialog.askopenfilename(title="Escolha imagem de referência",
                                       filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")])
        if f:
            self.ref_path.set(f)

    def run_stitch(self):
        if len(self.files) < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 imagens.")
            return
        t = threading.Thread(target=self._run_stitch_worker, daemon=True)
        self._toggle_running(True)
        t.start()

    def _run_stitch_worker(self):
        try:
            # Iniciar timer
            self.execution_start_time = time.time()
            self._update_timer()

            imgs = read_images(self.files)
            # Converter label para key
            label = self.method_label.get()
            key = self.label_to_key[label]
            fn = METHODS[key]["fn"]

            # Passa try_gpu apenas se o método realmente aceitar
            kwargs = {}
            if METHODS.get(key, {}).get("accepts_try_gpu", False):
                kwargs["try_gpu"] = self.try_gpu.get()

            if key == "classic/opencv_stitcher":
                pano = fn(imgs, modo=self.modo.get(), try_gpu=self.try_gpu.get())
            else:
                pano = fn(imgs, **kwargs) if kwargs else fn(imgs)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_key = key.replace("/", "-")
            out_path = self.output_dir / f"pano_{safe_key}_{ts}.jpg"
            ok = cv2.imwrite(str(out_path), pano)
            if not ok:
                raise RuntimeError("Falha ao salvar a imagem de saída.")

            # --- Sempre salvar métricas TXT com mesmo basename ---
            metrics_txt = out_path.with_suffix(".txt")
            metrics_data = {"psnr": float("nan"), "ssim": float("nan")}
            calc_done = False

            # Calcula métricas apenas se a usuária quiser (checkbox) e houver referência;
            # ainda assim o TXT é sempre criado (com NaN se não calcular).
            if self.do_metrics.get():
                if self.ref_path.get():
                    try:
                        self._log("📊 Calculando métricas vs referência...")
                        metrics_data = compute_metrics(ref_path=self.ref_path.get(), test_img=pano)
                        calc_done = True
                        self._log(f"✅ Métricas calculadas: PSNR={metrics_data.get('psnr', float('nan')):.3f} dB | SSIM={metrics_data.get('ssim', float('nan')):.4f}")
                    except Exception as met_err:
                        self._log(f"❌ Falha nas métricas: {str(met_err)}")
                        self.after(0, lambda: messagebox.showwarning("Erro nas Métricas",
                            f"Não foi possível calcular métricas:\n{str(met_err)}\n\nVerifique se a imagem de referência existe e é válida."))
                else:
                    self._log("⚠️  Checkbox de métricas marcado, mas nenhuma referência selecionada!")
                    self.after(0, lambda: messagebox.showwarning("Referência Necessária",
                        "Para calcular métricas, selecione uma imagem de referência (ground truth).\n\n"
                        "Clique em 'Escolher...' na seção 'Métricas (opcional)'."))

            save_metrics_txt(
                metrics_txt,
                key,
                ts,
                metrics_data,
                out_path.name,
                self.ref_path.get() or "(sem referência)"
            )
            append_metrics_log(self.output_dir, safe_key, ts, metrics_data, extra={"out": out_path.name})

            # Salvar último resultado e habilitar botão
            self.last_result_path = out_path
            self.after(0, lambda: self.open_result_btn.config(state=tk.NORMAL))

            # Calcular tempo total
            elapsed = time.time() - self.execution_start_time if self.execution_start_time else 0
            self._log(f"Concluído em {elapsed:.1f}s: {out_path.name}")

            # Salvar no histórico
            self._save_to_history(key, elapsed, True, str(out_path))

            self._notify_done(out_path)

            # Abrir a pasta de saída automaticamente
            try:
                import os, subprocess, platform
                folder = str(self.output_dir)
                if platform.system() == "Windows":
                    os.startfile(folder)  # type: ignore[attr-defined]
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception:
                pass

        except Exception as exc:
            msg = str(exc)
            self._log("Erro: " + msg)

            # Salvar erro no histórico
            label = self.method_label.get()
            key = self.label_to_key.get(label, "unknown")
            elapsed = time.time() - self.execution_start_time if self.execution_start_time else 0
            self._save_to_history(key, elapsed, False, "")

            self.after(0, lambda m=msg: messagebox.showerror("Erro no stitch", m))
        finally:
            # Parar timer
            self.execution_start_time = None
            self._toggle_running(False)

    def run_all_methods(self):
        if len(self.files) < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 imagens.")
            return
        t = threading.Thread(target=self._run_all_methods_worker, daemon=True)
        self._toggle_running(True)
        t.start()

    def _run_all_methods_worker(self):
        from datetime import datetime
        import shutil
        import pandas as pd

        try:
            # Criar pasta com timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_folder = self.output_dir / f"all_{ts}"
            all_folder.mkdir(parents=True, exist_ok=True)

            self._log(f"Executando todos os métodos... Salvando em: {all_folder.name}")

            # Verificar se há imagem de referência
            has_reference = bool(self.ref_path.get())
            if has_reference:
                self._log(f"Imagem de referência: {Path(self.ref_path.get()).name}")
                self._log("Métricas serão calculadas automaticamente para todos os métodos!")
            else:
                self._log("AVISO: Sem imagem de referência. Métricas PSNR/SSIM não serão calculadas.")

            imgs = read_images(self.files)

            # Ordenar métodos: classics primeiro, depois deep
            all_keys = list(METHODS.keys())
            classic_keys = sorted([k for k in all_keys if k.startswith("classic/")])
            deep_keys = sorted([k for k in all_keys if k.startswith("deep/")])
            ordered_keys = classic_keys + deep_keys

            # Todos os métodos corrigidos - não precisamos pular nenhum
            skip_methods = set()
            ordered_keys = [k for k in ordered_keys if k not in skip_methods]

            total = len(ordered_keys)
            success_count = 0
            error_count = 0
            skipped_count = len(skip_methods)

            # Lista para armazenar resultados das métricas
            resultados_metricas = []

            for idx, key in enumerate(ordered_keys, 1):
                method_label = METHODS[key]["label"]
                progress = int((idx / total) * 100)
                self.after(0, lambda p=progress: self.progress.config(value=p))
                self.after(0, lambda p=progress: self.progress_label.config(text=f"{p}%"))
                self._log(f"[{idx}/{total}] ({progress}%) Executando: {method_label}")

                tempo_inicio = time.time()
                resultado_metodo = {
                    'metodo': key.replace("/", "-"),
                    'label': method_label,
                    'tempo_segundos': None,
                    'psnr_db': None,
                    'ssim': None,
                    'mse': None,
                    'ghosting': None,
                    'seam_energy': None,
                    'overlap_ratio': None,
                    'deltaE_mean': None,
                    'deltaL_mean': None,
                    'entropia': None,
                    'nitidez': None,
                    'status': 'erro',
                    'arquivo': None,
                    'erro': None
                }

                try:
                    import signal
                    import multiprocessing
                    from multiprocessing import Process, Queue

                    def run_method_with_timeout(imgs, key, modo, try_gpu_val, result_queue):
                        """Executa método em processo separado"""
                        try:
                            fn = METHODS[key]["fn"]
                            kwargs = {}
                            if METHODS.get(key, {}).get("accepts_try_gpu", False):
                                kwargs["try_gpu"] = try_gpu_val

                            if key == "classic/opencv_stitcher":
                                pano = fn(imgs, modo=modo, try_gpu=try_gpu_val)
                            else:
                                pano = fn(imgs, **kwargs) if kwargs else fn(imgs)

                            result_queue.put(("success", pano))
                        except Exception as e:
                            result_queue.put(("error", str(e)))

                    # Timeout de 120 segundos por método
                    timeout = 120
                    result_queue = Queue()

                    # Executar em thread (Windows não suporta bem multiprocessing aqui)
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        fn = METHODS[key]["fn"]
                        kwargs = {}
                        if METHODS.get(key, {}).get("accepts_try_gpu", False):
                            kwargs["try_gpu"] = self.try_gpu.get()

                        if key == "classic/opencv_stitcher":
                            future = executor.submit(fn, imgs, modo=self.modo.get(), try_gpu=self.try_gpu.get())
                        else:
                            future = executor.submit(fn, imgs, **kwargs) if kwargs else executor.submit(fn, imgs)

                        try:
                            pano = future.result(timeout=timeout)
                        except concurrent.futures.TimeoutError:
                            raise RuntimeError(f"Timeout ({timeout}s excedido)")

                    # Calcular tempo
                    tempo_total = time.time() - tempo_inicio
                    resultado_metodo['tempo_segundos'] = tempo_total

                    # Salvar resultado
                    safe_key = key.replace("/", "-")
                    out_path = all_folder / f"pano_{safe_key}.jpg"
                    ok = cv2.imwrite(str(out_path), pano)
                    if not ok:
                        raise RuntimeError("Falha ao salvar a imagem de saída.")

                    resultado_metodo['arquivo'] = out_path.name
                    resultado_metodo['status'] = 'sucesso'

                    # Calcular métricas se houver referência
                    if has_reference:
                        try:
                            metrics_data = compute_metrics(ref_path=self.ref_path.get(), test_img=pano)

                            # Extrair todas as métricas do metrics_core
                            resultado_metodo['psnr_db'] = metrics_data.get('psnr')
                            resultado_metodo['ssim'] = metrics_data.get('ssim')
                            resultado_metodo['mse'] = metrics_data.get('mse')
                            resultado_metodo['ghosting'] = metrics_data.get('ghosting')
                            resultado_metodo['seam_energy'] = metrics_data.get('seam_energy')
                            resultado_metodo['overlap_ratio'] = metrics_data.get('overlap_ratio')
                            resultado_metodo['deltaE_mean'] = metrics_data.get('deltaE_mean')
                            resultado_metodo['deltaL_mean'] = metrics_data.get('deltaL_mean')

                            # Calcular métricas adicionais (entropia e nitidez)
                            import numpy as np
                            # Entropia (aproximada)
                            gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
                            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
                            hist = hist / hist.sum()
                            hist = hist[hist > 0]
                            entropia = -np.sum(hist * np.log2(hist))
                            resultado_metodo['entropia'] = entropia

                            # Nitidez (variância Laplaciano)
                            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                            resultado_metodo['nitidez'] = laplacian.var()

                            # Log com métricas principais incluindo ghosting
                            ghosting_val = resultado_metodo['ghosting']
                            ghosting_str = f" | Ghost={ghosting_val:.3f}" if ghosting_val is not None and not np.isnan(ghosting_val) else ""
                            seam_val = resultado_metodo['seam_energy']
                            seam_str = f" | Seam={seam_val:.2f}" if seam_val is not None and not np.isnan(seam_val) else ""

                            self._log(f"  ✓ Concluído: {out_path.name} | PSNR={metrics_data.get('psnr', 0):.2f} dB | SSIM={metrics_data.get('ssim', 0):.4f}{ghosting_str}{seam_str}")
                        except Exception as met_err:
                            self._log(f"  ✓ Concluído: {out_path.name} (erro nas métricas: {str(met_err)[:50]})")
                    else:
                        self._log(f"  ✓ Concluído: {out_path.name}")

                    success_count += 1

                except Exception as e:
                    error_msg = str(e).split('\n')[0][:100]  # Primeira linha, até 100 chars
                    self._log(f"  ✗ Erro: {error_msg}")
                    resultado_metodo['erro'] = error_msg
                    error_count += 1

                # Adicionar resultado à lista
                resultados_metricas.append(resultado_metodo)

            # Gerar CSV comparativo
            if resultados_metricas:
                csv_path = all_folder / "comparacao_metricas.csv"
                df = pd.DataFrame(resultados_metricas)

                # Ordenar colunas - incluindo todas as novas métricas
                colunas_ordem = ['metodo', 'label', 'status', 'tempo_segundos',
                                'psnr_db', 'ssim', 'mse',
                                'ghosting', 'seam_energy', 'overlap_ratio',
                                'deltaE_mean', 'deltaL_mean',
                                'entropia', 'nitidez',
                                'arquivo', 'erro']
                colunas_existentes = [c for c in colunas_ordem if c in df.columns]
                df = df[colunas_existentes]

                df.to_csv(csv_path, index=False, float_format='%.4f', encoding='utf-8-sig')
                self._log(f"\nCSV comparativo gerado: {csv_path.name}")

                # Gerar resumo estatístico
                if has_reference:
                    df_sucesso = df[df['status'] == 'sucesso']
                    if len(df_sucesso) > 0:
                        self._log("\nRESUMO DAS MÉTRICAS:")
                        if df_sucesso['psnr_db'].notna().any():
                            self._log(f"  PSNR médio: {df_sucesso['psnr_db'].mean():.2f} dB")
                            self._log(f"  PSNR melhor: {df_sucesso['psnr_db'].max():.2f} dB ({df_sucesso.loc[df_sucesso['psnr_db'].idxmax(), 'metodo']})")
                        if df_sucesso['ssim'].notna().any():
                            self._log(f"  SSIM médio: {df_sucesso['ssim'].mean():.4f}")
                            self._log(f"  SSIM melhor: {df_sucesso['ssim'].max():.4f} ({df_sucesso.loc[df_sucesso['ssim'].idxmax(), 'metodo']})")
                        if 'ghosting' in df_sucesso.columns and df_sucesso['ghosting'].notna().any():
                            self._log(f"  Ghosting médio: {df_sucesso['ghosting'].mean():.4f}")
                            self._log(f"  Ghosting melhor (menor): {df_sucesso['ghosting'].min():.4f} ({df_sucesso.loc[df_sucesso['ghosting'].idxmin(), 'metodo']})")
                        if 'seam_energy' in df_sucesso.columns and df_sucesso['seam_energy'].notna().any():
                            self._log(f"  Seam Energy médio: {df_sucesso['seam_energy'].mean():.2f}")
                            self._log(f"  Seam Energy melhor (menor): {df_sucesso['seam_energy'].min():.2f} ({df_sucesso.loc[df_sucesso['seam_energy'].idxmin(), 'metodo']})")

            # Resumo final
            self._log(f"\n{'='*50}")
            self._log(f"Concluído! Sucessos: {success_count}/{total}, Erros: {error_count}/{total}")
            self._log(f"Resultados salvos em: {all_folder}")
            if has_reference:
                self._log(f"CSV comparativo: {all_folder}/comparacao_metricas.csv")
            self._log(f"{'='*50}")

            # Abrir a pasta de saída automaticamente
            try:
                import os, subprocess, platform
                folder = str(all_folder)
                if platform.system() == "Windows":
                    os.startfile(folder)  # type: ignore[attr-defined]
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception:
                pass

            skip_msg = f"\nPulados: {skipped_count}" if skipped_count > 0 else ""
            self.after(0, lambda: messagebox.showinfo("Executar Todos",
                f"Concluído!\n\nSucessos: {success_count}/{total}\nErros: {error_count}/{total}{skip_msg}\n\nResultados em: {all_folder.name}"))

        except Exception as exc:
            msg = str(exc)
            self._log("Erro fatal: " + msg)
            self.after(0, lambda m=msg: messagebox.showerror("Erro ao executar todos", m))
        finally:
            self._toggle_running(False)

    def scan_folder(self):
        """Varre uma pasta raiz com subpastas contendo imagens e referências"""
        root_folder = filedialog.askdirectory(title="Escolha a pasta raiz para varrer")
        if not root_folder:
            return

        root_folder = Path(root_folder)

        # Verificar se há subpastas
        subfolders = [d for d in root_folder.iterdir() if d.is_dir()]
        if not subfolders:
            messagebox.showwarning("Aviso", "Nenhuma subpasta encontrada na pasta selecionada.")
            return

        self._log(f"Iniciando varredura de pasta: {root_folder.name}")
        self._log(f"Encontradas {len(subfolders)} subpastas")

        t = threading.Thread(target=self._scan_folder_worker, args=(root_folder,), daemon=True)
        self._toggle_running(True)
        t.start()

    def _scan_folder_worker(self, root_folder: Path):
        """Worker para varredura de pastas"""
        from datetime import datetime
        import pandas as pd

        try:
            # Criar pasta de resultados
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            scan_folder = self.output_dir / f"scan_{ts}"
            scan_folder.mkdir(parents=True, exist_ok=True)

            self._log(f"Resultados serão salvos em: {scan_folder.name}")

            # Encontrar todas as subpastas
            subfolders = sorted([d for d in root_folder.iterdir() if d.is_dir()])

            if not subfolders:
                self._log("ERRO: Nenhuma subpasta encontrada!")
                return

            # Lista para armazenar todos os resultados
            all_results = []

            total_folders = len(subfolders)

            for folder_idx, subfolder in enumerate(subfolders, 1):
                self._log(f"\n{'='*60}")
                self._log(f"[{folder_idx}/{total_folders}] Processando pasta: {subfolder.name}")
                self._log(f"{'='*60}")

                # Encontrar imagens numeradas e referência
                all_images = sorted([f for f in subfolder.iterdir()
                                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS])

                # Separar referência das imagens de entrada
                reference_img = None
                input_images = []

                for img in all_images:
                    if "reference" in img.stem.lower():
                        reference_img = img
                    else:
                        input_images.append(img)

                if not reference_img:
                    self._log(f"  ⚠️  AVISO: Nenhuma imagem 'reference' encontrada em {subfolder.name}")
                    self._log(f"  Pulando esta pasta...")
                    continue

                if len(input_images) < 2:
                    self._log(f"  ⚠️  AVISO: Menos de 2 imagens de entrada em {subfolder.name}")
                    self._log(f"  Pulando esta pasta...")
                    continue

                self._log(f"  Imagens de entrada: {len(input_images)}")
                self._log(f"  Imagem de referência: {reference_img.name}")

                # Ler as imagens
                try:
                    imgs = read_images(input_images)
                except Exception as e:
                    self._log(f"  ✗ Erro ao ler imagens: {str(e)}")
                    continue

                # Obter métodos ordenados
                all_keys = list(METHODS.keys())
                classic_keys = sorted([k for k in all_keys if k.startswith("classic/")])
                deep_keys = sorted([k for k in all_keys if k.startswith("deep/")])
                ordered_keys = classic_keys + deep_keys

                skip_methods = set()
                ordered_keys = [k for k in ordered_keys if k not in skip_methods]

                total_methods = len(ordered_keys)

                # Processar cada método
                for method_idx, key in enumerate(ordered_keys, 1):
                    method_label = METHODS[key]["label"]

                    # Atualizar progresso
                    overall_progress = int(((folder_idx - 1) * total_methods + method_idx) / (total_folders * total_methods) * 100)
                    self.after(0, lambda p=overall_progress: self.progress.config(value=p))
                    self.after(0, lambda p=overall_progress: self.progress_label.config(text=f"{p}%"))

                    self._log(f"  [{method_idx}/{total_methods}] {method_label}")

                    resultado = {
                        'pasta': subfolder.name,
                        'num_imagens': len(input_images),
                        'metodo': key.replace("/", "-"),
                        'label': method_label,
                        'tempo_segundos': None,
                        'psnr_db': None,
                        'ssim': None,
                        'mse': None,
                        'ghosting': None,
                        'seam_energy': None,
                        'overlap_ratio': None,
                        'deltaE_mean': None,
                        'deltaL_mean': None,
                        'entropia': None,
                        'nitidez': None,
                        'status': 'erro',
                        'arquivo': None,
                        'erro': None
                    }

                    try:
                        tempo_inicio = time.time()

                        # Executar método com timeout
                        import concurrent.futures
                        timeout = 120

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            fn = METHODS[key]["fn"]
                            kwargs = {}
                            if METHODS.get(key, {}).get("accepts_try_gpu", False):
                                kwargs["try_gpu"] = self.try_gpu.get()

                            if key == "classic/opencv_stitcher":
                                future = executor.submit(fn, imgs, modo=self.modo.get(), try_gpu=self.try_gpu.get())
                            else:
                                future = executor.submit(fn, imgs, **kwargs) if kwargs else executor.submit(fn, imgs)

                            try:
                                pano = future.result(timeout=timeout)
                            except concurrent.futures.TimeoutError:
                                raise RuntimeError(f"Timeout ({timeout}s excedido)")

                        tempo_total = time.time() - tempo_inicio
                        resultado['tempo_segundos'] = tempo_total

                        # Salvar panorama
                        safe_key = key.replace("/", "-")
                        out_filename = f"{subfolder.name}_{safe_key}.jpg"
                        out_path = scan_folder / out_filename
                        ok = cv2.imwrite(str(out_path), pano)
                        if not ok:
                            raise RuntimeError("Falha ao salvar imagem")

                        resultado['arquivo'] = out_filename
                        resultado['status'] = 'sucesso'

                        # Calcular métricas comparando com a referência local
                        try:
                            metrics_data = compute_metrics(ref_path=str(reference_img), test_img=pano)

                            # Extrair todas as métricas do metrics_core
                            resultado['psnr_db'] = metrics_data.get('psnr')
                            resultado['ssim'] = metrics_data.get('ssim')
                            resultado['mse'] = metrics_data.get('mse')
                            resultado['ghosting'] = metrics_data.get('ghosting')
                            resultado['seam_energy'] = metrics_data.get('seam_energy')
                            resultado['overlap_ratio'] = metrics_data.get('overlap_ratio')
                            resultado['deltaE_mean'] = metrics_data.get('deltaE_mean')
                            resultado['deltaL_mean'] = metrics_data.get('deltaL_mean')

                            # Calcular métricas adicionais (entropia e nitidez)
                            import numpy as np
                            gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
                            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
                            hist = hist / hist.sum()
                            hist = hist[hist > 0]
                            entropia = -np.sum(hist * np.log2(hist))
                            resultado['entropia'] = entropia

                            # Nitidez
                            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                            resultado['nitidez'] = laplacian.var()

                            # Log com métricas principais incluindo ghosting
                            ghosting_val = resultado['ghosting']
                            ghosting_str = f" | Ghost={ghosting_val:.3f}" if ghosting_val is not None and not np.isnan(ghosting_val) else ""
                            seam_val = resultado['seam_energy']
                            seam_str = f" | Seam={seam_val:.2f}" if seam_val is not None and not np.isnan(seam_val) else ""

                            self._log(f"    ✓ PSNR={resultado['psnr_db']:.2f} dB | SSIM={resultado['ssim']:.4f}{ghosting_str}{seam_str}")
                        except Exception as met_err:
                            self._log(f"    ✓ Salvo (erro nas métricas: {str(met_err)[:40]})")

                        # Gerar montagem before/after se opção estiver ativada
                        if self.do_before_after.get():
                            try:
                                # Criar pasta before_after dentro do scan_folder
                                before_after_folder = scan_folder / "before_after"
                                before_after_folder.mkdir(exist_ok=True)

                                # Nome do arquivo de montagem
                                montage_filename = f"{subfolder.name}_{safe_key}_comparison.jpg"
                                montage_path = before_after_folder / montage_filename

                                # Criar montagem
                                create_before_after_montage(
                                    input_imgs=input_images,
                                    ref_img_path=reference_img,
                                    result_img_path=out_path,
                                    output_path=montage_path
                                )
                                self._log(f"    📊 Montagem comparativa salva: {montage_filename}")
                            except Exception as montage_err:
                                self._log(f"    ⚠️  Erro ao gerar montagem: {str(montage_err)[:60]}")

                    except Exception as e:
                        error_msg = str(e).split('\n')[0][:80]
                        self._log(f"    ✗ Erro: {error_msg}")
                        resultado['erro'] = error_msg

                    all_results.append(resultado)

            # Gerar CSV consolidado
            if all_results:
                csv_path = scan_folder / "resultados_consolidados.csv"
                df = pd.DataFrame(all_results)

                # Ordenar colunas - incluindo todas as novas métricas
                colunas_ordem = ['pasta', 'num_imagens', 'metodo', 'label', 'status', 'tempo_segundos',
                               'psnr_db', 'ssim', 'mse',
                               'ghosting', 'seam_energy', 'overlap_ratio',
                               'deltaE_mean', 'deltaL_mean',
                               'entropia', 'nitidez',
                               'arquivo', 'erro']
                colunas_existentes = [c for c in colunas_ordem if c in df.columns]
                df = df[colunas_existentes]

                df.to_csv(csv_path, index=False, float_format='%.4f', encoding='utf-8-sig')
                self._log(f"\n{'='*60}")
                self._log(f"CSV consolidado gerado: {csv_path.name}")

                # Gerar estatísticas por pasta
                self._log(f"\nESTATÍSTICAS POR PASTA:")
                for pasta in df['pasta'].unique():
                    df_pasta = df[df['pasta'] == pasta]
                    df_sucesso = df_pasta[df_pasta['status'] == 'sucesso']
                    if len(df_sucesso) > 0:
                        self._log(f"\n  {pasta}:")
                        self._log(f"    Sucessos: {len(df_sucesso)}/{len(df_pasta)}")
                        if df_sucesso['psnr_db'].notna().any():
                            self._log(f"    PSNR médio: {df_sucesso['psnr_db'].mean():.2f} dB")
                            melhor_metodo = df_sucesso.loc[df_sucesso['psnr_db'].idxmax(), 'metodo']
                            self._log(f"    Melhor PSNR: {df_sucesso['psnr_db'].max():.2f} dB ({melhor_metodo})")
                        if df_sucesso['ssim'].notna().any():
                            self._log(f"    SSIM médio: {df_sucesso['ssim'].mean():.4f}")
                        if 'ghosting' in df_sucesso.columns and df_sucesso['ghosting'].notna().any():
                            self._log(f"    Ghosting médio: {df_sucesso['ghosting'].mean():.4f}")
                            melhor_metodo_ghost = df_sucesso.loc[df_sucesso['ghosting'].idxmin(), 'metodo']
                            self._log(f"    Melhor Ghosting (menor): {df_sucesso['ghosting'].min():.4f} ({melhor_metodo_ghost})")
                        if 'seam_energy' in df_sucesso.columns and df_sucesso['seam_energy'].notna().any():
                            self._log(f"    Seam Energy médio: {df_sucesso['seam_energy'].mean():.2f}")
                            melhor_metodo_seam = df_sucesso.loc[df_sucesso['seam_energy'].idxmin(), 'metodo']
                            self._log(f"    Melhor Seam Energy (menor): {df_sucesso['seam_energy'].min():.2f} ({melhor_metodo_seam})")

                self._log(f"\n{'='*60}")
                self._log(f"Varredura concluída!")
                self._log(f"Total de pastas processadas: {len(df['pasta'].unique())}")
                self._log(f"Total de resultados: {len(df)}")
                self._log(f"Resultados salvos em: {scan_folder}")
                self._log(f"{'='*60}")

                # Gerar pasta com top 5 métodos
                try:
                    self._generate_top5_report(df, scan_folder)
                except Exception as top5_err:
                    self._log(f"\n⚠️ Erro ao gerar relatório Top 5: {str(top5_err)}")

                # Abrir pasta de resultados
                try:
                    import os, subprocess, platform
                    folder = str(scan_folder)
                    if platform.system() == "Windows":
                        os.startfile(folder)  # type: ignore[attr-defined]
                    elif platform.system() == "Darwin":
                        subprocess.Popen(["open", folder])
                    else:
                        subprocess.Popen(["xdg-open", folder])
                except Exception:
                    pass

                self.after(0, lambda: messagebox.showinfo("Varredura Concluída",
                    f"Varredura concluída!\n\nPastas processadas: {len(df['pasta'].unique())}\n"
                    f"Total de resultados: {len(df)}\n\nResultados salvos em: {scan_folder.name}"))

        except Exception as exc:
            msg = str(exc)
            self._log(f"ERRO FATAL na varredura: {msg}")
            self.after(0, lambda m=msg: messagebox.showerror("Erro na varredura", m))
        finally:
            self._toggle_running(False)

    def _generate_top5_report(self, df, scan_folder: Path):
        """
        Gera pasta com os 5 melhores métodos baseado em análise multicriterial.
        Inclui montagens before/after e relatório detalhado.
        """
        import numpy as np
        import shutil
        from datetime import datetime

        self._log(f"\n{'='*60}")
        self._log("📊 GERANDO RELATÓRIO TOP 5 MÉTODOS")
        self._log(f"{'='*60}")

        # Filtrar apenas sucessos
        df_sucesso = df[df['status'] == 'sucesso'].copy()

        if len(df_sucesso) == 0:
            self._log("⚠️ Nenhum método teve sucesso. Não é possível gerar Top 5.")
            return

        # Verificar se há métricas disponíveis
        required_metrics = ['psnr_db', 'ssim']
        if not all(col in df_sucesso.columns for col in required_metrics):
            self._log("⚠️ Métricas PSNR/SSIM não disponíveis. Não é possível gerar Top 5.")
            return

        # Remover linhas com métricas NaN
        df_sucesso = df_sucesso.dropna(subset=['psnr_db', 'ssim'])

        if len(df_sucesso) == 0:
            self._log("⚠️ Nenhum método com métricas válidas. Não é possível gerar Top 5.")
            return

        # Calcular score normalizado ponderado
        self._log("\n🔍 Calculando scores multicriteriais...")

        # Normalizar PSNR (maior é melhor) - normalizar para 0-1
        psnr_min = df_sucesso['psnr_db'].min()
        psnr_max = df_sucesso['psnr_db'].max()
        if psnr_max > psnr_min:
            df_sucesso['psnr_norm'] = (df_sucesso['psnr_db'] - psnr_min) / (psnr_max - psnr_min)
        else:
            df_sucesso['psnr_norm'] = 0.5

        # Normalizar SSIM (maior é melhor, já está 0-1)
        df_sucesso['ssim_norm'] = df_sucesso['ssim']

        # Normalizar Ghosting (menor é melhor) - inverter escala
        if 'ghosting' in df_sucesso.columns:
            ghosting_valid = df_sucesso['ghosting'].notna()
            if ghosting_valid.any():
                ghost_min = df_sucesso.loc[ghosting_valid, 'ghosting'].min()
                ghost_max = df_sucesso.loc[ghosting_valid, 'ghosting'].max()
                if ghost_max > ghost_min:
                    df_sucesso.loc[ghosting_valid, 'ghosting_norm'] = 1 - (
                        (df_sucesso.loc[ghosting_valid, 'ghosting'] - ghost_min) / (ghost_max - ghost_min)
                    )
                else:
                    df_sucesso.loc[ghosting_valid, 'ghosting_norm'] = 0.5
                has_ghosting = True
            else:
                has_ghosting = False
        else:
            has_ghosting = False

        # Normalizar Seam Energy (menor é melhor) - inverter escala
        if 'seam_energy' in df_sucesso.columns:
            seam_valid = df_sucesso['seam_energy'].notna()
            if seam_valid.any():
                seam_min = df_sucesso.loc[seam_valid, 'seam_energy'].min()
                seam_max = df_sucesso.loc[seam_valid, 'seam_energy'].max()
                if seam_max > seam_min:
                    df_sucesso.loc[seam_valid, 'seam_norm'] = 1 - (
                        (df_sucesso.loc[seam_valid, 'seam_energy'] - seam_min) / (seam_max - seam_min)
                    )
                else:
                    df_sucesso.loc[seam_valid, 'seam_norm'] = 0.5
                has_seam = True
            else:
                has_seam = False
        else:
            has_seam = False

        # Calcular score final ponderado
        # Pesos: PSNR=30%, SSIM=30%, Ghosting=25%, Seam=15%
        peso_psnr = 0.30
        peso_ssim = 0.30
        peso_ghost = 0.25 if has_ghosting else 0.0
        peso_seam = 0.15 if has_seam else 0.0

        # Ajustar pesos se algumas métricas não existem
        total_peso = peso_psnr + peso_ssim + peso_ghost + peso_seam
        peso_psnr = peso_psnr / total_peso
        peso_ssim = peso_ssim / total_peso
        peso_ghost = peso_ghost / total_peso
        peso_seam = peso_seam / total_peso

        df_sucesso['score_final'] = (
            peso_psnr * df_sucesso['psnr_norm'] +
            peso_ssim * df_sucesso['ssim_norm']
        )

        if has_ghosting:
            df_sucesso['score_final'] += peso_ghost * df_sucesso['ghosting_norm'].fillna(0)

        if has_seam:
            df_sucesso['score_final'] += peso_seam * df_sucesso['seam_norm'].fillna(0)

        # Agrupar por método e calcular score médio
        df_metodos = df_sucesso.groupby('metodo').agg({
            'score_final': 'mean',
            'psnr_db': 'mean',
            'ssim': 'mean',
            'ghosting': 'mean' if has_ghosting else lambda x: np.nan,
            'seam_energy': 'mean' if has_seam else lambda x: np.nan,
            'tempo_segundos': 'mean',
            'status': 'count'  # número de sucessos
        }).reset_index()

        df_metodos.rename(columns={'status': 'num_sucessos'}, inplace=True)

        # Ordenar por score final
        df_metodos = df_metodos.sort_values('score_final', ascending=False)

        # Selecionar top 5
        top5 = df_metodos.head(5)

        self._log(f"\n🏆 TOP 5 MÉTODOS SELECIONADOS:")
        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            metodo_label = METHODS.get(row['metodo'].replace('-', '/'), {}).get('label', row['metodo'])
            self._log(f"{idx}. {metodo_label}")
            self._log(f"   Score: {row['score_final']:.4f}")
            self._log(f"   PSNR: {row['psnr_db']:.2f} dB | SSIM: {row['ssim']:.4f}")
            if has_ghosting and not np.isnan(row['ghosting']):
                self._log(f"   Ghosting: {row['ghosting']:.4f} | Seam Energy: {row['seam_energy']:.2f}")
            self._log(f"   Tempo médio: {row['tempo_segundos']:.2f}s | Sucessos: {int(row['num_sucessos'])}")

        # Criar pasta top_5_methods
        top5_folder = scan_folder / "top_5_methods"
        top5_folder.mkdir(exist_ok=True)

        # Copiar imagens e montagens
        self._log(f"\n📁 Copiando resultados dos Top 5 para: {top5_folder.name}")

        for idx, (_, row) in enumerate(top5.iterrows(), 1):
            metodo = row['metodo']
            metodo_key = metodo.replace('-', '/')
            metodo_label = METHODS.get(metodo_key, {}).get('label', metodo)

            # Obter todos os arquivos deste método
            metodo_results = df_sucesso[df_sucesso['metodo'] == metodo]

            for _, result_row in metodo_results.iterrows():
                pasta = result_row['pasta']
                arquivo = result_row['arquivo']

                if arquivo:
                    # Copiar imagem resultado
                    src_img = scan_folder / arquivo
                    if src_img.exists():
                        dst_img = top5_folder / f"rank{idx:02d}_{arquivo}"
                        shutil.copy2(src_img, dst_img)

                    # Copiar montagem before/after se existir
                    if self.do_before_after.get():
                        montage_name = arquivo.replace('.jpg', '_comparison.jpg')
                        src_montage = scan_folder / "before_after" / montage_name
                        if src_montage.exists():
                            dst_montage = top5_folder / f"rank{idx:02d}_{montage_name}"
                            shutil.copy2(src_montage, dst_montage)

        # Gerar relatório detalhado em TXT
        report_path = top5_folder / "RELATORIO_TOP5_METODOS.txt"
        self._log(f"\n📄 Gerando relatório detalhado: {report_path.name}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO: TOP 5 MÉTODOS DE IMAGE STITCHING\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Pasta de origem: {scan_folder.name}\n")
            f.write(f"Total de métodos avaliados: {len(df_metodos)}\n")
            f.write(f"Total de execuções bem-sucedidas: {len(df_sucesso)}\n\n")

            f.write("=" * 80 + "\n")
            f.write("METODOLOGIA DE SELEÇÃO\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. CRITÉRIOS DE AVALIAÇÃO\n\n")
            f.write("Os métodos foram avaliados com base em múltiplas métricas de qualidade:\n\n")

            f.write(f"   • PSNR (Peak Signal-to-Noise Ratio): {peso_psnr*100:.1f}%\n")
            f.write("     Mede a relação sinal/ruído. Valores maiores indicam menor ruído.\n")
            f.write("     Interpretação: >30dB=aceitável, 30-40dB=bom, >40dB=excelente\n\n")

            f.write(f"   • SSIM (Structural Similarity Index): {peso_ssim*100:.1f}%\n")
            f.write("     Avalia similaridade estrutural baseada em percepção humana.\n")
            f.write("     Interpretação: >0.90=excelente, 0.80-0.90=bom, <0.80=degradado\n\n")

            if has_ghosting:
                f.write(f"   • Ghosting Score: {peso_ghost*100:.1f}%\n")
                f.write("     Detecta artefatos de duplicação/sobreposição de bordas.\n")
                f.write("     Interpretação: <0.3=aceitável, 0.3-0.6=moderado, >0.6=severo\n")
                f.write("     (Valores menores são melhores)\n\n")

            if has_seam:
                f.write(f"   • Seam Energy: {peso_seam*100:.1f}%\n")
                f.write("     Mede energia de gradiente na zona de costura.\n")
                f.write("     (Valores menores indicam transições mais suaves)\n\n")

            f.write("\n2. PROCESSO DE NORMALIZAÇÃO E SCORING\n\n")
            f.write("Etapa 1: Todas as métricas foram normalizadas para escala 0-1\n")
            f.write("   - PSNR e SSIM: valores maiores são melhores (escala direta)\n")
            f.write("   - Ghosting e Seam: valores menores são melhores (escala invertida)\n\n")

            f.write("Etapa 2: Score final calculado com média ponderada:\n")
            f.write(f"   Score = {peso_psnr:.2f}×PSNR_norm + {peso_ssim:.2f}×SSIM_norm")
            if has_ghosting:
                f.write(f" + {peso_ghost:.2f}×Ghosting_norm")
            if has_seam:
                f.write(f" + {peso_seam:.2f}×Seam_norm")
            f.write("\n\n")

            f.write("Etapa 3: Para cada método, calculada média de scores em todos os datasets\n\n")

            f.write("Etapa 4: Métodos ordenados por score final (maior = melhor)\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("RANKING DOS TOP 5 MÉTODOS\n")
            f.write("=" * 80 + "\n\n")

            for idx, (_, row) in enumerate(top5.iterrows(), 1):
                metodo_key = row['metodo'].replace('-', '/')
                metodo_label = METHODS.get(metodo_key, {}).get('label', row['metodo'])

                f.write(f"\n{'─' * 80}\n")
                f.write(f"RANK #{idx}: {metodo_label}\n")
                f.write(f"{'─' * 80}\n\n")

                f.write(f"Score Final: {row['score_final']:.4f} / 1.0000\n\n")

                f.write("Métricas Médias:\n")
                f.write(f"  • PSNR: {row['psnr_db']:.2f} dB\n")
                f.write(f"  • SSIM: {row['ssim']:.4f}\n")

                if has_ghosting and not np.isnan(row['ghosting']):
                    f.write(f"  • Ghosting: {row['ghosting']:.4f}\n")
                if has_seam and not np.isnan(row['seam_energy']):
                    f.write(f"  • Seam Energy: {row['seam_energy']:.2f}\n")

                f.write(f"\nDesempenho:\n")
                f.write(f"  • Tempo médio: {row['tempo_segundos']:.2f} segundos\n")
                f.write(f"  • Taxa de sucesso: {int(row['num_sucessos'])} execuções bem-sucedidas\n")

                # Análise qualitativa
                f.write(f"\nAnálise:\n")

                # PSNR
                if row['psnr_db'] > 40:
                    f.write("  ✓ PSNR excelente (>40dB) - ruído praticamente imperceptível\n")
                elif row['psnr_db'] > 30:
                    f.write("  ✓ PSNR bom (30-40dB) - qualidade aceitável para aplicações práticas\n")
                else:
                    f.write("  ⚠ PSNR baixo (<30dB) - ruído pode ser perceptível\n")

                # SSIM
                if row['ssim'] > 0.90:
                    f.write("  ✓ SSIM excelente (>0.90) - preservação estrutural muito boa\n")
                elif row['ssim'] > 0.80:
                    f.write("  ✓ SSIM bom (0.80-0.90) - qualidade estrutural aceitável\n")
                else:
                    f.write("  ⚠ SSIM baixo (<0.80) - degradação estrutural perceptível\n")

                # Ghosting
                if has_ghosting and not np.isnan(row['ghosting']):
                    if row['ghosting'] < 0.3:
                        f.write("  ✓ Ghosting baixo (<0.3) - alinhamento preciso\n")
                    elif row['ghosting'] < 0.6:
                        f.write("  ⚠ Ghosting moderado (0.3-0.6) - artefatos perceptíveis\n")
                    else:
                        f.write("  ✗ Ghosting alto (>0.6) - desalinhamento significativo\n")

            f.write("\n\n" + "=" * 80 + "\n")
            f.write("RESUMO COMPARATIVO\n")
            f.write("=" * 80 + "\n\n")

            f.write("Método com melhor PSNR: ")
            best_psnr = df_metodos.loc[df_metodos['psnr_db'].idxmax()]
            best_psnr_label = METHODS.get(best_psnr['metodo'].replace('-', '/'), {}).get('label', best_psnr['metodo'])
            f.write(f"{best_psnr_label} ({best_psnr['psnr_db']:.2f} dB)\n\n")

            f.write("Método com melhor SSIM: ")
            best_ssim = df_metodos.loc[df_metodos['ssim'].idxmax()]
            best_ssim_label = METHODS.get(best_ssim['metodo'].replace('-', '/'), {}).get('label', best_ssim['metodo'])
            f.write(f"{best_ssim_label} ({best_ssim['ssim']:.4f})\n\n")

            if has_ghosting:
                f.write("Método com menor Ghosting: ")
                best_ghost = df_metodos.loc[df_metodos['ghosting'].idxmin()]
                best_ghost_label = METHODS.get(best_ghost['metodo'].replace('-', '/'), {}).get('label', best_ghost['metodo'])
                f.write(f"{best_ghost_label} ({best_ghost['ghosting']:.4f})\n\n")

            f.write("Método mais rápido: ")
            fastest = df_metodos.loc[df_metodos['tempo_segundos'].idxmin()]
            fastest_label = METHODS.get(fastest['metodo'].replace('-', '/'), {}).get('label', fastest['metodo'])
            f.write(f"{fastest_label} ({fastest['tempo_segundos']:.2f}s)\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("CONCLUSÕES E RECOMENDAÇÕES\n")
            f.write("=" * 80 + "\n\n")

            top1 = top5.iloc[0]
            top1_label = METHODS.get(top1['metodo'].replace('-', '/'), {}).get('label', top1['metodo'])

            f.write(f"O método '{top1_label}' obteve o melhor score geral,\n")
            f.write(f"com pontuação de {top1['score_final']:.4f}, demonstrando o melhor equilíbrio\n")
            f.write("entre qualidade visual (PSNR/SSIM) e precisão de alinhamento (Ghosting/Seam).\n\n")

            f.write("RECOMENDAÇÕES DE USO:\n\n")
            for idx, (_, row) in enumerate(top5.iterrows(), 1):
                metodo_label = METHODS.get(row['metodo'].replace('-', '/'), {}).get('label', row['metodo'])
                if idx == 1:
                    f.write(f"{idx}. {metodo_label}:\n")
                    f.write("   → Recomendado para uso geral. Melhor equilíbrio qualidade/precisão.\n\n")
                elif row['tempo_segundos'] < top5['tempo_segundos'].quantile(0.5):
                    f.write(f"{idx}. {metodo_label}:\n")
                    f.write("   → Alternativa rápida mantendo boa qualidade.\n\n")
                else:
                    f.write(f"{idx}. {metodo_label}:\n")
                    f.write("   → Boa alternativa quando qualidade é prioritária.\n\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("RANKING COMPLETO - TODOS OS MÉTODOS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Lista completa dos {len(df_metodos)} métodos avaliados, ordenados por score final.\n")
            f.write("Esta seção fornece uma visão panorâmica de todos os métodos testados.\n\n")

            f.write("Legenda:\n")
            f.write("  • Score: pontuação multicriterial final (0-1, maior=melhor)\n")
            f.write("  • PSNR: Peak Signal-to-Noise Ratio em dB (maior=melhor)\n")
            f.write("  • SSIM: Structural Similarity Index 0-1 (maior=melhor)\n")
            if has_ghosting:
                f.write("  • Ghost: Ghosting Score 0-1 (menor=melhor)\n")
            if has_seam:
                f.write("  • Seam: Seam Energy (menor=melhor)\n")
            f.write("  • Tempo: tempo médio de execução em segundos\n")
            f.write("  • N: número de execuções bem-sucedidas\n\n")

            f.write("-" * 80 + "\n")
            f.write(f"{'#':<4} {'Método':<45} {'Score':<8} {'PSNR':<7} {'SSIM':<7}\n")
            f.write("-" * 80 + "\n")

            for idx, (_, row) in enumerate(df_metodos.iterrows(), 1):
                metodo_key = row['metodo'].replace('-', '/')
                metodo_label = METHODS.get(metodo_key, {}).get('label', row['metodo'])

                # Truncar label se muito longo
                if len(metodo_label) > 44:
                    metodo_label = metodo_label[:41] + "..."

                # Indicador de Top 5
                top_indicator = "🏆" if idx <= 5 else "  "

                f.write(f"{idx:<3}{top_indicator} {metodo_label:<45} {row['score_final']:.4f}   "
                       f"{row['psnr_db']:6.2f}  {row['ssim']:.4f}\n")

            f.write("-" * 80 + "\n\n")

            # Tabela detalhada completa
            f.write("\nDETALHES COMPLETOS DE CADA MÉTODO:\n")
            f.write("=" * 80 + "\n\n")

            for idx, (_, row) in enumerate(df_metodos.iterrows(), 1):
                metodo_key = row['metodo'].replace('-', '/')
                metodo_label = METHODS.get(metodo_key, {}).get('label', row['metodo'])

                # Separador visual
                if idx <= 5:
                    f.write(f"\n{'━' * 80}\n")
                    f.write(f"🏆 RANK #{idx} (TOP 5): {metodo_label}\n")
                    f.write(f"{'━' * 80}\n")
                else:
                    f.write(f"\n{'─' * 80}\n")
                    f.write(f"RANK #{idx}: {metodo_label}\n")
                    f.write(f"{'─' * 80}\n")

                f.write(f"\nScore Final: {row['score_final']:.4f}\n")
                f.write(f"PSNR: {row['psnr_db']:.2f} dB | SSIM: {row['ssim']:.4f}")

                if has_ghosting and not np.isnan(row['ghosting']):
                    f.write(f" | Ghosting: {row['ghosting']:.4f}")
                if has_seam and not np.isnan(row['seam_energy']):
                    f.write(f" | Seam: {row['seam_energy']:.2f}")

                f.write(f"\nTempo médio: {row['tempo_segundos']:.2f}s | Sucessos: {int(row['num_sucessos'])}")

                # Classificação qualitativa resumida
                f.write("\n")

                # PSNR
                if row['psnr_db'] > 40:
                    psnr_qual = "excelente"
                elif row['psnr_db'] > 30:
                    psnr_qual = "bom"
                else:
                    psnr_qual = "regular"

                # SSIM
                if row['ssim'] > 0.90:
                    ssim_qual = "excelente"
                elif row['ssim'] > 0.80:
                    ssim_qual = "bom"
                else:
                    ssim_qual = "regular"

                f.write(f"Qualidade: PSNR {psnr_qual}, SSIM {ssim_qual}")

                # Ghosting
                if has_ghosting and not np.isnan(row['ghosting']):
                    if row['ghosting'] < 0.3:
                        ghost_qual = "baixo"
                    elif row['ghosting'] < 0.6:
                        ghost_qual = "moderado"
                    else:
                        ghost_qual = "alto"
                    f.write(f", Ghosting {ghost_qual}")

                f.write("\n")

            # Estatísticas gerais
            f.write("\n\n" + "=" * 80 + "\n")
            f.write("ESTATÍSTICAS GERAIS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total de métodos avaliados: {len(df_metodos)}\n")
            f.write(f"Total de execuções bem-sucedidas: {len(df_sucesso)}\n\n")

            f.write("Estatísticas de PSNR:\n")
            f.write(f"  • Média geral: {df_metodos['psnr_db'].mean():.2f} dB\n")
            f.write(f"  • Mediana: {df_metodos['psnr_db'].median():.2f} dB\n")
            f.write(f"  • Melhor: {df_metodos['psnr_db'].max():.2f} dB\n")
            f.write(f"  • Pior: {df_metodos['psnr_db'].min():.2f} dB\n")
            f.write(f"  • Desvio padrão: {df_metodos['psnr_db'].std():.2f} dB\n\n")

            f.write("Estatísticas de SSIM:\n")
            f.write(f"  • Média geral: {df_metodos['ssim'].mean():.4f}\n")
            f.write(f"  • Mediana: {df_metodos['ssim'].median():.4f}\n")
            f.write(f"  • Melhor: {df_metodos['ssim'].max():.4f}\n")
            f.write(f"  • Pior: {df_metodos['ssim'].min():.4f}\n")
            f.write(f"  • Desvio padrão: {df_metodos['ssim'].std():.4f}\n\n")

            if has_ghosting:
                ghosting_stats = df_metodos['ghosting'].dropna()
                if len(ghosting_stats) > 0:
                    f.write("Estatísticas de Ghosting:\n")
                    f.write(f"  • Média geral: {ghosting_stats.mean():.4f}\n")
                    f.write(f"  • Mediana: {ghosting_stats.median():.4f}\n")
                    f.write(f"  • Melhor (menor): {ghosting_stats.min():.4f}\n")
                    f.write(f"  • Pior (maior): {ghosting_stats.max():.4f}\n")
                    f.write(f"  • Desvio padrão: {ghosting_stats.std():.4f}\n\n")

            f.write("Estatísticas de Tempo de Execução:\n")
            f.write(f"  • Média geral: {df_metodos['tempo_segundos'].mean():.2f} s\n")
            f.write(f"  • Mediana: {df_metodos['tempo_segundos'].median():.2f} s\n")
            f.write(f"  • Mais rápido: {df_metodos['tempo_segundos'].min():.2f} s\n")
            f.write(f"  • Mais lento: {df_metodos['tempo_segundos'].max():.2f} s\n\n")

            # Distribuição por categoria
            f.write("Distribuição por Categoria:\n")
            classicos = len([m for m in df_metodos['metodo'] if m.startswith('classic')])
            deep = len([m for m in df_metodos['metodo'] if m.startswith('deep')])
            f.write(f"  • Métodos Clássicos: {classicos}\n")
            f.write(f"  • Métodos Deep Learning: {deep}\n\n")

            # Top 5 por categoria
            if classicos > 0:
                df_classic = df_metodos[df_metodos['metodo'].str.startswith('classic')].head(3)
                f.write("Top 3 Métodos Clássicos:\n")
                for idx, (_, row) in enumerate(df_classic.iterrows(), 1):
                    label = METHODS.get(row['metodo'].replace('-', '/'), {}).get('label', row['metodo'])
                    f.write(f"  {idx}. {label} (Score: {row['score_final']:.4f})\n")
                f.write("\n")

            if deep > 0:
                df_deep = df_metodos[df_metodos['metodo'].str.startswith('deep')].head(3)
                f.write("Top 3 Métodos Deep Learning:\n")
                for idx, (_, row) in enumerate(df_deep.iterrows(), 1):
                    label = METHODS.get(row['metodo'].replace('-', '/'), {}).get('label', row['metodo'])
                    f.write(f"  {idx}. {label} (Score: {row['score_final']:.4f})\n")
                f.write("\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("=" * 80 + "\n")

        self._log(f"✓ Relatório gerado com sucesso!")
        self._log(f"✓ {len(top5)} métodos copiados para: {top5_folder.name}")
        self._log(f"{'='*60}\n")

    # ---------- Utils ----------
    def _toggle_running(self, running: bool):
        def set_state():
            self.run_btn.config(state=(tk.DISABLED if running else tk.NORMAL))
            self.run_all_btn.config(state=(tk.DISABLED if running else tk.NORMAL))
            self.scan_folder_btn.config(state=(tk.DISABLED if running else tk.NORMAL))
            if not running:
                self.progress.config(value=0)
                self.progress_label.config(text="0%")
        self.after(0, set_state)

    def _log(self, msg: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.configure(state=tk.DISABLED)
        self.log.see(tk.END)

    def _notify_done(self, out_path: Path):
        self.after(0, lambda: messagebox.showinfo("Concluído", "Panorama salvo em: " + str(out_path)))

    def _update_timer(self):
        """Atualiza o timer a cada segundo durante execução"""
        if self.execution_start_time:
            elapsed = time.time() - self.execution_start_time
            mins, secs = divmod(int(elapsed), 60)
            self.time_var.set(f"{mins:02d}:{secs:02d}")
            self.after(1000, self._update_timer)  # Atualizar a cada 1s
        else:
            self.time_var.set("--:--")

    def _setup_shortcuts(self):
        """Configura atalhos de teclado"""
        self.bind("<Control-o>", lambda e: self.add_files())
        self.bind("<Control-O>", lambda e: self.add_files())
        self.bind("<Control-r>", lambda e: self.run_stitch())
        self.bind("<Control-R>", lambda e: self.run_stitch())
        self.bind("<F5>", lambda e: self.run_all_methods())
        self.bind("<Delete>", lambda e: self.remove_selected())
        self.bind("<Control-l>", lambda e: self.clear_files())
        self.bind("<Control-L>", lambda e: self.clear_files())
        self.bind("<Control-s>", lambda e: self.save_preset())
        self.bind("<Control-S>", lambda e: self.save_preset())
        self.bind("<Control-p>", lambda e: self.load_preset())
        self.bind("<Control-P>", lambda e: self.load_preset())
        self.bind("<Control-h>", lambda e: self.show_history())
        self.bind("<Control-H>", lambda e: self.show_history())
        self.bind("<Control-e>", lambda e: self.export_batch_report())
        self.bind("<Control-E>", lambda e: self.export_batch_report())
        self.bind("<F1>", lambda e: self.show_help())

    def open_result(self):
        """Abre o último resultado gerado"""
        if self.last_result_path and self.last_result_path.exists():
            import os, subprocess, platform
            try:
                file_path = str(self.last_result_path)
                if platform.system() == "Windows":
                    os.startfile(file_path)  # type: ignore
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", file_path])
                else:
                    subprocess.Popen(["xdg-open", file_path])
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível abrir: {e}")
        else:
            messagebox.showinfo("Info", "Nenhum resultado disponível ainda.")

    def rerun_last(self):
        """Re-executa o último método"""
        if not self.last_result_path or not self.files:
            messagebox.showwarning("Aviso", "Nenhuma execução anterior encontrada")
            return
        self.run_stitch()

    def _build_status_bar(self):
        """Cria a barra de status no rodapé"""
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _update_status(self):
        """Atualiza a barra de status e contador de imagens"""
        count = len(self.files)
        method = self.method_label.get()
        status = f"📁 {count} {'imagem' if count == 1 else 'imagens'} | 🔧 {method} | 💾 {self.output_dir.name} | ✓ Pronto"
        self.status_var.set(status)
        self.img_count_label.config(text=f"{count} {'imagem' if count == 1 else 'imagens'}")

        # Habilitar/desabilitar botão de re-executar
        if self.last_result_path:
            self.rerun_btn.config(state=tk.NORMAL)

    def _show_preview(self):
        """Mostra thumbnail da primeira imagem selecionada"""
        if not self.files:
            self.preview_label.config(image="", text="Selecione imagens...")
            if hasattr(self.preview_label, 'image'):
                self.preview_label.image = None
            return

        try:
            img = Image.open(self.files[0])
            img.thumbnail((200, 150))
            photo = ImageTk.PhotoImage(img)
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo  # Mantém referência
        except Exception as e:
            self.preview_label.config(text=f"Erro ao carregar preview")

    def _on_method_change(self, event=None):
        """Atualiza a status bar quando o método é alterado"""
        label = self.method_label.get()
        key = self.label_to_key.get(label, "")
        desc = METHOD_DESCRIPTIONS.get(key, "Método de stitching de imagens")

        count = len(self.files)
        status = f"📁 {count} {'imagem' if count == 1 else 'imagens'} | 🔧 {desc}"
        self.status_var.set(status)

    def _setup_drag_drop(self):
        """Configura drag & drop de arquivos (requer tkinterdnd2)"""
        try:
            from tkinterdnd2 import DND_FILES, TkinterDnD

            def drop(event):
                files = self.tk.splitlist(event.data)
                new_files = [Path(f) for f in files if Path(f).suffix.lower() in SUPPORTED_EXTS]
                if new_files:
                    self._append_files(new_files)

            # Registrar drag & drop na listbox
            self.listbox.drop_target_register(DND_FILES)
            self.listbox.dnd_bind('<<Drop>>', drop)
            self._log("Drag & Drop habilitado!")
        except ImportError:
            # Se tkinterdnd2 não estiver disponível, apenas ignore
            self._log("Dica: Instale tkinterdnd2 para drag & drop: pip install tkinterdnd2")
        except Exception:
            pass

    def save_preset(self):
        """Salva configurações atuais como preset"""
        from tkinter import simpledialog
        name = simpledialog.askstring("Salvar Preset", "Nome do preset:")
        if not name:
            return

        preset = {
            "method": self.method_label.get(),
            "mode": self.modo.get(),
            "gpu": self.try_gpu.get(),
            "do_metrics": self.do_metrics.get(),
            "ref_path": self.ref_path.get(),
            "output_dir": str(self.output_dir)
        }

        self.presets[name] = preset
        presets_file = Path(__file__).parent / "presets.json"
        presets_file.write_text(json.dumps(self.presets, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("Sucesso", f"Preset '{name}' salvo!")
        self._log(f"Preset '{name}' salvo com sucesso.")

    def load_preset(self):
        """Carrega preset salvo"""
        from tkinter import simpledialog
        if not self.presets:
            messagebox.showinfo("Info", "Nenhum preset salvo ainda.")
            return

        # Mostrar diálogo de seleção
        preset_names = list(self.presets.keys())
        name = simpledialog.askstring("Carregar Preset",
                                      f"Presets disponíveis:\n{', '.join(preset_names)}\n\nNome do preset:")
        if not name or name not in self.presets:
            return

        preset = self.presets[name]
        self.method_label.set(preset.get("method", self.method_label.get()))
        self.modo.set(preset.get("mode", "panorama"))
        self.try_gpu.set(preset.get("gpu", False))
        self.do_metrics.set(preset.get("do_metrics", False))
        self.ref_path.set(preset.get("ref_path", ""))

        out_dir = preset.get("output_dir", str(self.output_dir))
        self.output_dir = Path(out_dir)
        self.out_dir_var.set(out_dir)

        messagebox.showinfo("Sucesso", f"Preset '{name}' carregado!")
        self._log(f"Preset '{name}' carregado.")
        self._update_status()

    def _load_presets(self):
        """Carrega presets salvos do disco"""
        presets_file = Path(__file__).parent / "presets.json"
        if presets_file.exists():
            try:
                self.presets = json.loads(presets_file.read_text(encoding="utf-8"))
            except Exception:
                self.presets = {}

    def _load_history(self):
        """Carrega histórico de execuções"""
        history_file = self.output_dir / "history.json"
        if history_file.exists():
            try:
                self.history = json.loads(history_file.read_text(encoding="utf-8"))
            except Exception:
                self.history = []

    def _save_to_history(self, key: str, elapsed: float, success: bool, output_path: str = ""):
        """Salva execução no histórico"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "method": key,
            "elapsed": elapsed,
            "success": success,
            "output": output_path,
            "num_images": len(self.files)
        }
        self.history.append(entry)

        # Manter apenas últimas 100 execuções
        self.history = self.history[-100:]

        history_file = self.output_dir / "history.json"
        history_file.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")

    def show_history(self):
        """Mostra janela com histórico de execuções"""
        if not self.history:
            messagebox.showinfo("Histórico", "Nenhuma execução registrada ainda.")
            return

        win = tk.Toplevel(self)
        win.title("Histórico de Execuções")
        win.geometry("800x500")

        # Criar treeview
        columns = ("Data/Hora", "Método", "Tempo (s)", "Status", "Imagens")
        tree = ttk.Treeview(win, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150)

        # Preencher dados (últimas 50)
        for entry in reversed(self.history[-50:]):
            timestamp = entry.get("timestamp", "")
            method = entry.get("method", "")
            elapsed = f"{entry.get('elapsed', 0):.1f}"
            status = "✓ Sucesso" if entry.get("success", False) else "✗ Erro"
            num_imgs = entry.get("num_images", 0)

            tree.insert("", 0, values=(timestamp, method, elapsed, status, num_imgs))

        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Botão para exportar
        btn_frame = ttk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Exportar CSV", command=lambda: self._export_history_csv()).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="Fechar", command=win.destroy).pack(side=tk.RIGHT)

    def _export_history_csv(self):
        """Exporta histórico para CSV"""
        if not self.history:
            return

        csv_path = self.output_dir / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Timestamp,Method,Elapsed(s),Success,NumImages,Output\n")
            for entry in self.history:
                f.write(f"{entry.get('timestamp', '')},{entry.get('method', '')},"
                       f"{entry.get('elapsed', 0):.2f},{entry.get('success', False)},"
                       f"{entry.get('num_images', 0)},{entry.get('output', '')}\n")

        messagebox.showinfo("Exportado", f"Histórico salvo em:\n{csv_path.name}")
        self._log(f"Histórico exportado: {csv_path.name}")

    def export_batch_report(self):
        """Exporta relatório completo de execução em batch"""
        # Verificar se existe pasta all_*
        all_folders = sorted([d for d in self.output_dir.glob("all_*") if d.is_dir()], reverse=True)

        if not all_folders:
            messagebox.showinfo("Info", "Nenhuma execução em batch encontrada.\nExecute 'Todos' primeiro.")
            return

        # Usar a pasta mais recente
        latest_folder = all_folders[0]

        # Coletar resultados
        results = []
        for img_path in sorted(latest_folder.glob("pano_*.jpg")):
            method = img_path.stem.replace("pano_", "")
            size = img_path.stat().st_size / (1024 * 1024)  # MB
            results.append({"method": method, "file": img_path.name, "size_mb": f"{size:.2f}"})

        if not results:
            messagebox.showinfo("Info", "Nenhum resultado encontrado na pasta batch.")
            return

        # Criar relatório HTML
        html_path = latest_folder / "report.html"
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Batch Stitching Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .summary {{ background: white; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>🖼️ Batch Stitching Report</h1>
    <div class="summary">
        <p><strong>Pasta:</strong> {latest_folder.name}</p>
        <p><strong>Total de métodos:</strong> {len(results)}</p>
        <p><strong>Data/Hora:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    <table>
        <tr>
            <th>#</th>
            <th>Método</th>
            <th>Arquivo</th>
            <th>Tamanho (MB)</th>
        </tr>
"""
        for idx, r in enumerate(results, 1):
            html_content += f"""
        <tr>
            <td>{idx}</td>
            <td>{r['method']}</td>
            <td>{r['file']}</td>
            <td>{r['size_mb']}</td>
        </tr>
"""

        html_content += """
    </table>
</body>
</html>
"""

        html_path.write_text(html_content, encoding='utf-8')
        messagebox.showinfo("Relatório Criado", f"Relatório HTML salvo em:\n{html_path.name}")
        self._log(f"Relatório batch criado: {html_path.name}")

        # Abrir relatório no navegador
        try:
            import webbrowser
            webbrowser.open(str(html_path))
        except Exception:
            pass

    def show_help(self):
        """Mostra janela de ajuda com atalhos"""
        help_text = """
🔧 ATALHOS DE TECLADO:

Ctrl+O       Adicionar imagens
Ctrl+R       Executar stitch
F5           Executar todos os métodos
Delete       Remover selecionados
Ctrl+L       Limpar lista

Ctrl+S       Salvar preset
Ctrl+P       Carregar preset
Ctrl+H       Ver histórico
Ctrl+E       Exportar relatório batch
F1           Mostrar esta ajuda

📁 DRAG & DROP:
Arraste imagens diretamente para a lista!

💡 DICAS:
- Use "Executar Todos" (F5) para comparar métodos
- Salve configurações favoritas como presets
- Histórico rastreia todas execuções
- Veja descrição do método na barra de status
"""
        messagebox.showinfo("Ajuda - Atalhos", help_text)

# ---------- Métricas ----------
def compute_metrics(ref_path: str, test_img):
    """
    Calcula métricas completas comparando test_img com referência.
    Retorna: psnr, ssim, ghosting, seam_energy, overlap_ratio, deltaE_mean, deltaL_mean, mse

    IMPORTANTE: Usa máscara para ignorar bordas pretas (áreas sem conteúdo) do panorama.
    """
    import numpy as np, cv2

    # Tentar usar metrics_core completo
    try:
        from modules.metrics.metrics_core import compute_all, align_pair
        ref = cv2.imread(ref_path)
        if ref is None:
            raise RuntimeError("Não consegui ler a referência: " + ref_path)

        # Alinhar tamanhos usando método pad (preserva proporções)
        # Isso garante que não haja deformação da imagem de referência
        if ref.shape[:2] != test_img.shape[:2]:
            ref, test_img_aligned = align_pair(ref, test_img, how="pad")
        else:
            test_img_aligned = test_img

        # Usar compute_all com use_mask=True para ignorar bordas pretas
        # Isso garante que apenas a região válida do panorama seja comparada
        metrics = compute_all(ref, test_img_aligned, use_mask=True)

        # Renomear chaves para minúsculas consistentes
        return {
            'psnr': metrics.get('PSNR', float('nan')),
            'ssim': metrics.get('SSIM', float('nan')),
            'mse': metrics.get('MSE', float('nan')),
            'ghosting': metrics.get('ghosting', float('nan')),
            'seam_energy': metrics.get('seam_energy', float('nan')),
            'overlap_ratio': metrics.get('overlap_ratio', float('nan')),
            'deltaE_mean': metrics.get('deltaE_mean', float('nan')),
            'deltaL_mean': metrics.get('deltaL_mean', float('nan')),
            'L_ref': metrics.get('L_ref', float('nan')),
            'L_cmp': metrics.get('L_cmp', float('nan'))
        }
    except Exception as e:
        # Fallback para método antigo se metrics_core não estiver disponível
        ref = cv2.imread(ref_path)
        if ref is None:
            raise RuntimeError("Não consegui ler a referência: " + ref_path)
        if ref.shape[:2] != test_img.shape[:2]:
            ref = cv2.resize(ref, (test_img.shape[1], test_img.shape[0]))

        psnr = float(cv2.PSNR(ref, test_img))
        def _ssim(a,b):
            a,b = a.astype('float64'), b.astype('float64')
            C1=(0.01*255)**2; C2=(0.03*255)**2
            mu_a,mu_b=a.mean(),b.mean(); va, vb = a.var(), b.var()
            cab=((a-mu_a)*(b-mu_b)).mean()
            return ((2*mu_a*mu_b + C1)*(2*cab + C2))/((mu_a**2 + mu_b**2 + C1)*(va + vb + C2))
        ssim = float((_ssim(test_img[:,:,0],ref[:,:,0]) + _ssim(test_img[:,:,1],ref[:,:,1]) + _ssim(test_img[:,:,2],ref[:,:,2]))/3.0)

        return {
            'psnr': psnr,
            'ssim': ssim,
            'mse': float('nan'),
            'ghosting': float('nan'),
            'seam_energy': float('nan'),
            'overlap_ratio': float('nan'),
            'deltaE_mean': float('nan'),
            'deltaL_mean': float('nan'),
            'L_ref': float('nan'),
            'L_cmp': float('nan')
        }

if __name__ == "__main__":
    app = StitcherGUI()
    app.mainloop()
