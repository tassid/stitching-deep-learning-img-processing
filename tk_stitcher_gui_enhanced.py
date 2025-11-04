# -*- coding: utf-8 -*-
"""
Enhanced Stitcher GUI - Versão Completa
Interface gráfica com TODAS as funcionalidades:
- 20+ métodos de stitching (clássicos + deep learning)
- Detecção YOLO (carros/pessoas)
- Stitching direcional e grid
- Processamento de vídeo
- Run All Methods + Scan Folder + Top 5 Analysis
- Sistema completo de métricas (PSNR, SSIM, ghosting, seam energy, etc.)
- Presets e histórico de execução
- Relatórios CSV/HTML
- Atalhos de teclado
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import json
import time
import cv2
import numpy as np
from PIL import Image, ImageTk

# ===== Constantes =====
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ST_OK = getattr(cv2, "STITCHER_OK", 0)
ST_ERR_NEED_MORE_IMGS = getattr(cv2, "STITCHER_ERR_NEED_MORE_IMGS", 1)
ST_ERR_HOMOGRAPHY_EST_FAIL = getattr(cv2, "STITCHER_ERR_HOMOGRAPHY_EST_FAIL", 2)
ST_ERR_CAMERA_PARAMS_ADJUST_FAIL = getattr(cv2, "STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL", 3)

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

# ===== Utility Functions =====
def status_to_msg(status: int) -> str:
    mapping = {
        ST_ERR_NEED_MORE_IMGS: "Não há imagens suficientes (>=2) ou sobreposição insuficiente.",
        ST_ERR_HOMOGRAPHY_EST_FAIL: "Falha ao estimar homografia (poucas features ou não correspondem).",
        ST_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Falha ao ajustar parâmetros da câmera.",
    }
    tip = " Dicas: mantenha ordem/seq, 30-50% de overlap, iluminação semelhante."
    return mapping.get(status, "Falhou com status " + str(status) + ".") + tip

def make_stitcher(modo: str, try_gpu: bool):
    mode = cv2.Stitcher_PANORAMA if modo == "panorama" else cv2.Stitcher_SCANS
    st = cv2.Stitcher_create(mode)
    if hasattr(st, "setPanoConfidenceThresh"):
        st.setPanoConfidenceThresh(0.5)
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

def save_metrics_json(path: Path, metrics: dict):
    Path(path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

def create_before_after_montage(input_imgs: List[Path], ref_img_path: Path, result_img_path: Path, output_path: Path, max_height: int = 400):
    """Cria montagem comparativa Before/After"""
    def resize_keep_aspect(img, max_h):
        h, w = img.shape[:2]
        if h > max_h:
            scale = max_h / h
            new_w = int(w * scale)
            return cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
        return img

    def add_label(img, text, color=(255, 255, 255)):
        h, w = img.shape[:2]
        img_copy = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (10, h - text_h - 20), (20 + text_w, h - 10), (0, 0, 0), -1)
        img_copy = cv2.addWeighted(overlay, 0.6, img_copy, 0.4, 0)
        cv2.putText(img_copy, text, (15, h - 15), font, font_scale, color, thickness, cv2.LINE_AA)
        return img_copy

    # Carregar imagens originais
    originals = []
    for p in input_imgs:
        img = cv2.imread(str(p))
        if img is not None:
            img = resize_keep_aspect(img, max_height)
            originals.append(img)

    # Carregar referência e resultado
    ref_img = cv2.imread(str(ref_img_path))
    result_img = cv2.imread(str(result_img_path))

    if ref_img is None or result_img is None:
        raise RuntimeError("Erro ao carregar imagens para montagem")

    ref_img = resize_keep_aspect(ref_img, max_height)
    result_img = resize_keep_aspect(result_img, max_height)

    # Adicionar labels
    for i, img in enumerate(originals):
        originals[i] = add_label(img, f"Input {i+1:02d}")
    ref_img = add_label(ref_img, "Reference")
    result_img = add_label(result_img, "Result")

    # Criar montagem
    row1 = np.hstack(originals)
    row2 = np.hstack([ref_img, result_img])

    # Ajustar larguras
    max_width = max(row1.shape[1], row2.shape[1])
    if row1.shape[1] < max_width:
        padding = np.ones((row1.shape[0], max_width - row1.shape[1], 3), dtype=np.uint8) * 255
        row1 = np.hstack([row1, padding])
    if row2.shape[1] < max_width:
        padding = np.ones((row2.shape[0], max_width - row2.shape[1], 3), dtype=np.uint8) * 255
        row2 = np.hstack([row2, padding])

    montage = np.vstack([row1, row2])
    cv2.imwrite(str(output_path), montage)
    return montage

# ===== Metrics Computation =====
def compute_metrics(ref_path: str, test_img):
    """Calcula métricas completas usando metrics_core"""
    try:
        from modules.metrics.metrics_core import compute_all, align_pair
        ref = cv2.imread(ref_path)
        if ref is None:
            raise RuntimeError("Erro ao ler referência: " + ref_path)

        if ref.shape[:2] != test_img.shape[:2]:
            ref, test_img_aligned = align_pair(ref, test_img, how="pad")
        else:
            test_img_aligned = test_img

        metrics = compute_all(ref, test_img_aligned, use_mask=True)

        return {
            'psnr': metrics.get('PSNR', float('nan')),
            'ssim': metrics.get('SSIM', float('nan')),
            'mse': metrics.get('MSE', float('nan')),
            'ghosting': metrics.get('ghosting', float('nan')),
            'seam_energy': metrics.get('seam_energy', float('nan')),
            'overlap_ratio': metrics.get('overlap_ratio', float('nan')),
            'deltaE_mean': metrics.get('deltaE_mean', float('nan')),
            'deltaL_mean': metrics.get('deltaL_mean', float('nan')),
        }
    except Exception as e:
        # Fallback para PSNR/SSIM básicos
        ref = cv2.imread(ref_path)
        if ref is None:
            raise RuntimeError("Erro ao ler referência: " + ref_path)
        if ref.shape[:2] != test_img.shape[:2]:
            ref = cv2.resize(ref, (test_img.shape[1], test_img.shape[0]))

        psnr = float(cv2.PSNR(ref, test_img))
        # SSIM básico
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
        }

# ===== Method Registry =====
METHODS = {}

METHOD_DESCRIPTIONS = {
    "classic/1_stitcher_basico": "SIFT + homografia simples. Base sólida e rápida.",
    "classic/2_manual_sift_ransac_feather": "SIFT manual com feather blending suave.",
    "classic/3_features_homography": "Features clássicos (SIFT/ORB) com múltiplas opções.",
    "classic/4_blend_exposure": "Feather blend suave baseado em distância.",
    "classic/5_blend_wavelet": "Fusão por wavelets. Preserva detalhes finos.",
    "classic/6_seams_exposure_multiband": "Graph-cut seams + multiband. Resultado profissional.",
    "classic/7_detectors_harris_fast": "Detectores Harris/FAST. Leve e rápido.",
    "classic/8_blend_laplacian": "Pirâmide Laplaciana. Suave e natural.",
    "classic/9_exposure_comp_only": "Compensação de exposição + feather.",
    "classic/10_blend_poisson": "Seamless cloning de Poisson. Remove costuras.",
    "classic/11_blend_multifocal": "Blend baseado em foco.",
    "classic/12_blend_hybrid": "Combina contraste, saturação e exposição.",
    "classic/13_ecc_alignment": "Enhanced Correlation Coefficient.",
    "classic/opencv_stitcher": "Stitcher oficial OpenCV. Rápido e confiável.",
    "classic/orb_homography": "ORB + RANSAC. Bom para pouca textura.",
    "deep/deep_homography": "CNN para homografia. Robusto.",
    "deep/flow_stitch": "SIFT robusto com fallback.",
    "deep/loftr": "Detector-free matching. Ótimo para pouca textura.",
    "deep/sp_superglue": "SuperPoint + SuperGlue. Robusto e versátil.",
    "deep/udis": "SIFT simplificado. Rápido.",
}

def register_method(key, label, description=""):
    def deco(fn):
        METHODS[key] = {"label": label, "fn": fn, "accepts_try_gpu": True, "description": description}
        return fn
    return deco

def discover_algorithms():
    """Descobre automaticamente todos os algoritmos em modules/classic e modules/deep"""
    import importlib, pkgutil, sys, tempfile, shutil, subprocess, inspect

    def add_bridge(mod_name: str, key: str, label: str):
        # Tentar import direto
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

        # Subprocesso bridge
        def _bridge(imgs, **kwargs):
            tmp = tempfile.mkdtemp(prefix=key.replace("/", "_") + "_")
            try:
                in_dir = Path(tmp) / "imgs"
                in_dir.mkdir(parents=True, exist_ok=True)
                for i, im in enumerate(imgs):
                    cv2.imwrite(str(in_dir / f"im_{i:03d}.jpg"), im)
                out_file = Path(tmp) / "out.jpg"
                cmd = [sys.executable, "-m", mod_name, "--dir", str(in_dir), "--out", str(out_file)]
                if kwargs.get("try_gpu", False):
                    try:
                        import torch
                        if torch.cuda.is_available():
                            cmd += ["--device", "cuda"]
                    except Exception:
                        pass
                ret = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if ret.returncode != 0:
                    raise RuntimeError(f"{label} falhou\n{ret.stderr}")
                pano = cv2.imread(str(out_file))
                if pano is None:
                    raise RuntimeError(label + " não gerou imagem")
                return pano
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        METHODS[key] = {"label": label, "fn": _bridge, "accepts_try_gpu": True}

    # Descobrir clássicos
    try:
        import modules.classic as classic_pkg
        for info in pkgutil.iter_modules(classic_pkg.__path__):
            name = info.name
            if name.startswith("__"):
                continue
            clean_name = name
            if name[0].isdigit():
                parts = name.split("_", 1)
                clean_name = parts[1] if len(parts) > 1 else name
            add_bridge(f"modules.classic.{name}", f"classic/{name}", "Clássico • " + clean_name.replace("_", " ").title())
    except Exception:
        pass

    # Descobrir deep learning
    try:
        import modules.deep as deep_pkg
        skip_modules = {"harmonize_cnn", "harmonize_gan"}
        for info in pkgutil.iter_modules(deep_pkg.__path__):
            name = info.name
            if name.startswith("__") or name in skip_modules:
                continue
            add_bridge(f"modules.deep.{name}", f"deep/{name}", "Deep • " + name.replace("_", " ").title())
    except Exception:
        pass

    # Métodos nativos
    METHODS.setdefault("classic/opencv_stitcher", {"label": "Clássico • OpenCV Stitcher", "fn": method_opencv_stitcher, "accepts_try_gpu": False})
    METHODS.setdefault("classic/orb_homography", {"label": "Clássico • ORB Homography", "fn": method_orb_homography, "accepts_try_gpu": False})

@register_method("classic/opencv_stitcher", "Clássico • OpenCV Stitcher")
def method_opencv_stitcher(imgs, modo="panorama", try_gpu=False):
    st = make_stitcher(modo, try_gpu)
    status, pano = st.stitch(imgs)
    if status != ST_OK:
        raise RuntimeError(status_to_msg(status))
    return pano

@register_method("classic/orb_homography", "Clássico • ORB Homography")
def method_orb_homography(imgs, ratio=0.75, ransac_reproj=4.0, try_gpu: bool=False):
    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    base = imgs[0]
    for next_img in imgs[1:]:
        kp1, des1 = orb.detectAndCompute(base, None)
        kp2, des2 = orb.detectAndCompute(next_img, None)
        if des1 is None or des2 is None:
            raise RuntimeError("ORB não detectou features suficientes")

        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < ratio * n.distance:
                    good.append(m)

        if len(good) < 10:
            raise RuntimeError("Poucas correspondências entre imagens")

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, ransac_reproj)
        if H is None:
            raise RuntimeError("Falha ao calcular homografia")

        h1, w1 = base.shape[:2]
        h2, w2 = next_img.shape[:2]

        corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)

        all_corners = np.concatenate([
            np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
            transformed
        ], axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

        result = cv2.warpPerspective(next_img, translation @ H, (x_max - x_min, y_max - y_min))
        result[-y_min:-y_min + h1, -x_min:-x_min + w1] = base

        base = result

    return base

# Importar módulos adicionais
try:
    from modules.detection import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from modules.advanced_stitching import GridStitcher, DirectionalStitcher
    ADVANCED_STITCHING_AVAILABLE = True
except ImportError:
    ADVANCED_STITCHING_AVAILABLE = False

try:
    from modules.video import VideoProcessor
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False


# ===== Main GUI =====
class EnhancedStitcherGUI(tk.Tk):
    """GUI Completo com todas as funcionalidades"""

    def __init__(self):
        super().__init__()
        self.title("Stitcher — Complete Edition")
        self.geometry("1400x800")
        self.minsize(1200, 700)

        # Descobrir algoritmos
        discover_algorithms()

        # Variáveis principais
        self.files: List[Path] = []
        self.yolo_files: List[Path] = []
        self.output_dir = Path(__file__).resolve().parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Variáveis de controle
        self.running = False
        self.execution_start_time = None
        self.last_output_path = None
        self.video_path = tk.StringVar(value="")

        # Módulos especializados
        self.yolo_detector: Optional[YOLODetector] = None
        self.video_processor: Optional[VideoProcessor] = None

        # Construir interface
        self._build_notebook()
        self._build_status_bar()
        self._setup_keyboard_shortcuts()

        # Carregar histórico e presets
        self.history_file = self.output_dir / "history.json"
        self.presets_file = self.output_dir / "presets.json"
        self.history = self._load_history()
        self.presets = self._load_presets()

    def _build_notebook(self):
        """Cria notebook com todas as abas"""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Aba 1: Stitching Clássico com TODOS os métodos
        self.tab_classic = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_classic, text="🔧 Stitching Completo")
        self._build_classic_tab()

        # Aba 2: Grid & Direcional
        if ADVANCED_STITCHING_AVAILABLE:
            self.tab_advanced = ttk.Frame(self.notebook)
            self.notebook.add(self.tab_advanced, text="🎯 Grid & Direcional")
            self._build_advanced_tab()

        # Aba 3: Detecção YOLO
        if YOLO_AVAILABLE:
            self.tab_yolo = ttk.Frame(self.notebook)
            self.notebook.add(self.tab_yolo, text="🚗 Detecção YOLO")
            self._build_yolo_tab()

        # Aba 4: Vídeo
        if VIDEO_AVAILABLE:
            self.tab_video = ttk.Frame(self.notebook)
            self.notebook.add(self.tab_video, text="🎬 Vídeo")
            self._build_video_tab()

    def _build_classic_tab(self):
        """Aba completa de stitching com TODOS os métodos"""
        main_frame = ttk.Frame(self.tab_classic)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame esquerdo - Lista de imagens
        left_frame = ttk.LabelFrame(main_frame, text="📁 Imagens")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Botões de arquivo
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="➕ Adicionar", command=self.add_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="📁 Pasta", command=self.add_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑 Limpar", command=self.clear_files).pack(side=tk.LEFT, padx=2)

        # Listbox
        list_container = ttk.Frame(left_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.listbox = tk.Listbox(list_container, selectmode=tk.EXTENDED)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.img_count_label = ttk.Label(left_frame, text="0 imagens")
        self.img_count_label.pack(padx=10, pady=(0, 10))

        # Frame direito - Controles
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Seleção de método
        method_frame = ttk.LabelFrame(right_frame, text="⚙ Método")
        method_frame.pack(fill=tk.X, pady=(0, 10))

        # Preparar lista de métodos
        self.method_labels = []
        self.label_to_key = {}

        # Ordenar: clássicos primeiro, depois deep
        classic_keys = sorted([k for k in METHODS.keys() if k.startswith("classic/")])
        deep_keys = sorted([k for k in METHODS.keys() if k.startswith("deep/")])

        for key in classic_keys + deep_keys:
            label = METHODS[key]["label"]
            self.method_labels.append(label)
            self.label_to_key[label] = key

        self.method_var = tk.StringVar(value=self.method_labels[0] if self.method_labels else "")
        method_combo = ttk.Combobox(method_frame, textvariable=self.method_var,
                                    values=self.method_labels, state="readonly", width=30)
        method_combo.pack(padx=10, pady=10)

        # Modo
        mode_frame = ttk.LabelFrame(right_frame, text="📐 Modo")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        self.modo = tk.StringVar(value="panorama")
        ttk.Radiobutton(mode_frame, text="Panorama", variable=self.modo, value="panorama").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(mode_frame, text="Scans", variable=self.modo, value="scans").pack(anchor=tk.W, padx=10, pady=3)

        # GPU
        gpu_frame = ttk.Frame(right_frame)
        gpu_frame.pack(fill=tk.X, pady=(0, 10))
        self.try_gpu = tk.BooleanVar(value=False)
        ttk.Checkbutton(gpu_frame, text="🚀 Usar GPU", variable=self.try_gpu).pack(anchor=tk.W, padx=10)

        # Métricas
        metrics_frame = ttk.LabelFrame(right_frame, text="📊 Métricas")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))

        self.calc_metrics = tk.BooleanVar(value=False)
        ttk.Checkbutton(metrics_frame, text="Calcular métricas", variable=self.calc_metrics).pack(anchor=tk.W, padx=10, pady=5)

        ref_frame = ttk.Frame(metrics_frame)
        ref_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(ref_frame, text="Referência:").pack(side=tk.LEFT)
        self.ref_path = tk.StringVar(value="")
        ttk.Entry(ref_frame, textvariable=self.ref_path, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(ref_frame, text="📂", command=self.select_reference, width=3).pack(side=tk.LEFT)

        self.create_montage = tk.BooleanVar(value=False)
        ttk.Checkbutton(metrics_frame, text="Montagem Before/After", variable=self.create_montage).pack(anchor=tk.W, padx=10, pady=5)

        # Output dir
        out_frame = ttk.LabelFrame(right_frame, text="📂 Saída")
        out_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(out_frame, text=str(self.output_dir.name), wraplength=250).pack(padx=10, pady=5)
        ttk.Button(out_frame, text="Trocar pasta", command=self.change_output_dir).pack(padx=10, pady=5)

        # Botões de ação
        action_frame = ttk.Frame(right_frame)
        action_frame.pack(fill=tk.X, pady=10)

        self.run_btn = ttk.Button(action_frame, text="▶ Executar", command=self.run_stitch, width=20)
        self.run_btn.pack(pady=3)
        self.run_all_btn = ttk.Button(action_frame, text="⚡ Executar Todos (F5)", command=self.run_all_methods, width=20)
        self.run_all_btn.pack(pady=3)
        self.scan_folder_btn = ttk.Button(action_frame, text="🔍 Scan Folder", command=self.scan_folder, width=20)
        self.scan_folder_btn.pack(pady=3)
        ttk.Button(action_frame, text="📊 Histórico", command=self.show_history, width=20).pack(pady=3)
        ttk.Button(action_frame, text="📂 Abrir Resultados", command=self.open_results_folder, width=20).pack(pady=3)

        # Presets
        preset_frame = ttk.Frame(right_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        ttk.Button(preset_frame, text="💾 Salvar Preset", command=self.save_preset, width=13).pack(side=tk.LEFT, padx=2)
        ttk.Button(preset_frame, text="📥 Carregar", command=self.load_preset, width=13).pack(side=tk.LEFT, padx=2)

        # Log console
        log_frame = ttk.LabelFrame(right_frame, text="📋 Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log = tk.Text(log_frame, height=12, width=50, state=tk.DISABLED, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5, padx=(0, 5))

    def _build_advanced_tab(self):
        """Aba Grid & Direcional"""
        frame = ttk.Frame(self.tab_advanced)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Stitching Avançado", font=("", 14, "bold")).pack(pady=(0, 15))

        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Direcional
        left_col = ttk.LabelFrame(main_container, text="🔄 Stitching Direcional")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.direction_var = tk.StringVar(value="horizontal")
        ttk.Radiobutton(left_col, text="→ Horizontal", variable=self.direction_var, value="horizontal").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(left_col, text="↓ Vertical", variable=self.direction_var, value="vertical").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Button(left_col, text="▶ Executar Direcional", command=self.run_directional, width=25).pack(pady=15)

        # Grid
        right_col = ttk.LabelFrame(main_container, text="⊞ Stitching em Grid")
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.grid_var = tk.StringVar(value="2x2")
        ttk.Radiobutton(right_col, text="2x2 (4 imagens)", variable=self.grid_var, value="2x2").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(right_col, text="3x3 (9 imagens)", variable=self.grid_var, value="3x3").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(right_col, text="2x3 (6 imagens)", variable=self.grid_var, value="2x3").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Radiobutton(right_col, text="Auto", variable=self.grid_var, value="auto").pack(anchor=tk.W, padx=10, pady=5)
        ttk.Button(right_col, text="▶ Executar Grid", command=self.run_grid, width=25).pack(pady=15)

        self.advanced_status = ttk.Label(frame, text="Pronto", relief=tk.SUNKEN, anchor=tk.W)
        self.advanced_status.pack(fill=tk.X, pady=(10, 0))

    def _build_yolo_tab(self):
        """Aba YOLO (mantém implementação anterior)"""
        frame = ttk.Frame(self.tab_yolo)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Detecção de Objetos com YOLO", font=("", 14, "bold")).pack(pady=(0, 15))

        # Estrutura similar à versão anterior...
        main_container = ttk.Frame(frame)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Coluna esquerda - Seleção
        left_col = ttk.Frame(main_container)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        selection_frame = ttk.LabelFrame(left_col, text="📁 Imagens")
        selection_frame.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(selection_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text="📂 Pasta", command=self.yolo_select_folder).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="➕ Adicionar", command=self.yolo_add_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="🗑 Limpar", command=self.yolo_clear_files).pack(side=tk.LEFT, padx=2)

        list_container = ttk.Frame(selection_frame)
        list_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.yolo_listbox = tk.Listbox(list_container, selectmode=tk.EXTENDED, height=8)
        yolo_scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.yolo_listbox.yview)
        self.yolo_listbox.configure(yscrollcommand=yolo_scrollbar.set)
        self.yolo_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yolo_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.yolo_img_count = ttk.Label(selection_frame, text="0 imagens")
        self.yolo_img_count.pack(padx=10, pady=(0, 10))

        # Coluna direita - Config
        right_col = ttk.Frame(main_container)
        right_col.pack(side=tk.RIGHT, fill=tk.Y)

        config_frame = ttk.LabelFrame(right_col, text="⚙ Modelo")
        config_frame.pack(fill=tk.X, pady=(0, 10))

        version_frame = ttk.Frame(config_frame)
        version_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(version_frame, text="Versão:").pack(side=tk.LEFT)
        self.yolo_version = tk.StringVar(value="yolov8")
        ttk.Combobox(version_frame, textvariable=self.yolo_version,
                    values=["yolov5", "yolov8", "yolov11"],
                    state="readonly", width=12).pack(side=tk.LEFT, padx=5)

        size_frame = ttk.Frame(config_frame)
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(size_frame, text="Tamanho:").pack(side=tk.LEFT)
        self.yolo_size = tk.StringVar(value="n")
        ttk.Combobox(size_frame, textvariable=self.yolo_size,
                    values=["n", "s", "m", "l", "x"],
                    state="readonly", width=12).pack(side=tk.LEFT, padx=5)

        device_frame = ttk.Frame(config_frame)
        device_frame.pack(fill=tk.X, padx=10, pady=5)
        self.yolo_device = tk.StringVar(value="cuda")
        ttk.Checkbutton(device_frame, text="Usar GPU", variable=self.yolo_device,
                       onvalue="cuda", offvalue="cpu").pack(side=tk.LEFT)

        ttk.Button(config_frame, text="🔄 Carregar Modelo", command=self.load_yolo_model).pack(pady=10)

        # Opções
        detect_frame = ttk.LabelFrame(frame, text="🎯 Detecção")
        detect_frame.pack(fill=tk.X, pady=(0, 10))

        self.detect_what = tk.StringVar(value="both")
        ttk.Radiobutton(detect_frame, text="🚗 Carros", variable=self.detect_what, value="cars").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(detect_frame, text="👤 Pessoas", variable=self.detect_what, value="persons").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(detect_frame, text="🚗👤 Ambos", variable=self.detect_what, value="both").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(detect_frame, text="🌐 Todos", variable=self.detect_what, value="all").pack(anchor=tk.W, padx=10, pady=3)

        conf_frame = ttk.Frame(detect_frame)
        conf_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(conf_frame, text="Confiança:").pack(side=tk.LEFT)
        self.yolo_conf = tk.DoubleVar(value=0.25)
        ttk.Scale(conf_frame, from_=0.1, to=0.9, variable=self.yolo_conf, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.conf_label = ttk.Label(conf_frame, text="0.25")
        self.conf_label.pack(side=tk.LEFT)
        self.yolo_conf.trace_add("write", lambda *_: self.conf_label.config(text=f"{self.yolo_conf.get():.2f}"))

        # Ações
        action_frame = ttk.Frame(frame)
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="▶ Detectar em Imagens", command=self.run_yolo_detection, width=25).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="▶ Detectar em Panorama", command=self.run_yolo_on_panorama, width=25).pack(side=tk.LEFT, padx=5)

        self.yolo_status = ttk.Label(frame, text="Modelo não carregado", relief=tk.SUNKEN, anchor=tk.W)
        self.yolo_status.pack(fill=tk.X, pady=(10, 0))

    def _build_video_tab(self):
        """Aba de vídeo (mantém implementação anterior)"""
        frame = ttk.Frame(self.tab_video)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="Processamento de Vídeo", font=("", 14, "bold")).pack(pady=(0, 15))

        video_frame = ttk.LabelFrame(frame, text="📹 Vídeo")
        video_frame.pack(fill=tk.X, pady=(0, 10))

        path_frame = ttk.Frame(video_frame)
        path_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Entry(path_frame, textvariable=self.video_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_frame, text="📂", command=self.select_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(path_frame, text="ℹ", command=self.show_video_info).pack(side=tk.LEFT)

        extract_frame = ttk.LabelFrame(frame, text="⚙ Extração")
        extract_frame.pack(fill=tk.X, pady=(0, 10))

        self.extract_method = tk.StringVar(value="interval")
        ttk.Radiobutton(extract_frame, text="Por intervalo", variable=self.extract_method, value="interval").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(extract_frame, text="Por tempo", variable=self.extract_method, value="time").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(extract_frame, text="Keyframes", variable=self.extract_method, value="keyframes").pack(anchor=tk.W, padx=10, pady=3)

        param_frame = ttk.Frame(extract_frame)
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(param_frame, text="Intervalo:").pack(side=tk.LEFT)
        self.video_interval = tk.IntVar(value=10)
        ttk.Spinbox(param_frame, from_=1, to=100, textvariable=self.video_interval, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(param_frame, text="Máx. frames:").pack(side=tk.LEFT, padx=(15, 0))
        self.video_max_frames = tk.IntVar(value=20)
        ttk.Spinbox(param_frame, from_=1, to=1000, textvariable=self.video_max_frames, width=10).pack(side=tk.LEFT, padx=5)

        action_frame = ttk.LabelFrame(frame, text="🎬 Ações")
        action_frame.pack(fill=tk.X, pady=(0, 10))

        self.video_action = tk.StringVar(value="save")
        ttk.Radiobutton(action_frame, text="💾 Salvar frames", variable=self.video_action, value="save").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(action_frame, text="🎯 Salvar + Stitching", variable=self.video_action, value="stitch").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(action_frame, text="🚗 Salvar + YOLO", variable=self.video_action, value="detect").pack(anchor=tk.W, padx=10, pady=3)
        ttk.Radiobutton(action_frame, text="⊞ Salvar + Grid", variable=self.video_action, value="grid").pack(anchor=tk.W, padx=10, pady=3)

        ttk.Button(frame, text="▶ Processar Vídeo", command=self.run_video_processing, width=30).pack(pady=15)

        self.video_status = ttk.Label(frame, text="Pronto", relief=tk.SUNKEN, anchor=tk.W)
        self.video_status.pack(fill=tk.X, pady=(10, 0))

    def _build_status_bar(self):
        """Barra de status e progresso"""
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(5, 10))

        self.status_label = ttk.Label(status_frame, text="Pronto", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress = ttk.Progressbar(status_frame, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=(10, 0))

        self.progress_label = ttk.Label(status_frame, text="0%", width=5)
        self.progress_label.pack(side=tk.RIGHT, padx=(10, 5))

        self.timer_label = ttk.Label(status_frame, text="00:00", width=6)
        self.timer_label.pack(side=tk.RIGHT)

    def _setup_keyboard_shortcuts(self):
        """Configura atalhos de teclado"""
        self.bind("<Control-o>", lambda e: self.add_files())
        self.bind("<Control-r>", lambda e: self.run_stitch())
        self.bind("<F5>", lambda e: self.run_all_methods())
        self.bind("<Delete>", lambda e: self.remove_selected())
        self.bind("<Control-l>", lambda e: self.clear_files())
        self.bind("<Control-s>", lambda e: self.save_preset())
        self.bind("<Control-p>", lambda e: self.load_preset())
        self.bind("<Control-h>", lambda e: self.show_history())
        self.bind("<F1>", lambda e: self.show_help())

    # ===== File Management =====
    def add_files(self):
        """Adiciona imagens"""
        paths = filedialog.askopenfilenames(
            title="Selecionar imagens",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("Todos", "*.*")]
        )
        for p in paths:
            path = Path(p)
            if path not in self.files:
                self.files.append(path)
                self.listbox.insert(tk.END, path.name)
        self.img_count_label.config(text=f"{len(self.files)} imagens")

    def add_folder(self):
        """Adiciona pasta"""
        folder = filedialog.askdirectory(title="Selecionar pasta")
        if not folder:
            return
        folder_path = Path(folder)
        for ext in SUPPORTED_EXTS:
            for img in folder_path.glob(f"*{ext}"):
                if img not in self.files:
                    self.files.append(img)
                    self.listbox.insert(tk.END, img.name)
        self.img_count_label.config(text=f"{len(self.files)} imagens")

    def clear_files(self):
        """Limpa lista"""
        self.files.clear()
        self.listbox.delete(0, tk.END)
        self.img_count_label.config(text="0 imagens")

    def remove_selected(self):
        """Remove selecionados"""
        selection = self.listbox.curselection()
        for idx in reversed(selection):
            self.files.pop(idx)
            self.listbox.delete(idx)
        self.img_count_label.config(text=f"{len(self.files)} imagens")

    def select_reference(self):
        """Seleciona imagem de referência"""
        path = filedialog.askopenfilename(
            title="Imagem de referência",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")]
        )
        if path:
            self.ref_path.set(path)

    def change_output_dir(self):
        """Troca diretório de saída"""
        folder = filedialog.askdirectory(title="Pasta de saída")
        if folder:
            self.output_dir = Path(folder)

    def open_results_folder(self):
        """Abre pasta de resultados"""
        import os, subprocess, platform
        if platform.system() == 'Windows':
            os.startfile(str(self.output_dir))
        elif platform.system() == 'Darwin':
            subprocess.Popen(['open', str(self.output_dir)])
        else:
            subprocess.Popen(['xdg-open', str(self.output_dir)])

    # ===== Classic Stitching =====
    def run_stitch(self):
        """Executa método selecionado"""
        if len(self.files) < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 imagens")
            return

        self.status_label.config(text="Executando...")
        self.running = True
        self.execution_start_time = time.time()
        self._update_timer()

        t = threading.Thread(target=self._run_stitch_worker, daemon=True)
        t.start()

    def _run_stitch_worker(self):
        """Worker de stitching"""
        try:
            imgs = read_images(self.files)

            label = self.method_var.get()
            key = self.label_to_key.get(label, "classic/opencv_stitcher")

            fn = METHODS[key]["fn"]
            kwargs = {}
            if METHODS[key].get("accepts_try_gpu", False):
                kwargs["try_gpu"] = self.try_gpu.get()

            # Executar método
            if key == "classic/opencv_stitcher":
                pano = fn(imgs, modo=self.modo.get(), try_gpu=self.try_gpu.get())
            else:
                pano = fn(imgs, **kwargs) if kwargs else fn(imgs)

            # Salvar
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_key = key.replace("/", "-")
            out_path = self.output_dir / f"pano_{safe_key}_{ts}.jpg"
            cv2.imwrite(str(out_path), pano)
            self.last_output_path = out_path

            # Métricas
            if self.calc_metrics.get() and self.ref_path.get():
                try:
                    metrics = compute_metrics(self.ref_path.get(), pano)
                    json_path = out_path.with_suffix('.json')
                    save_metrics_json(json_path, metrics)

                    # Montagem
                    if self.create_montage.get():
                        montage_path = out_path.parent / f"montage_{out_path.stem}.jpg"
                        create_before_after_montage(self.files, Path(self.ref_path.get()), out_path, montage_path)

                    self.status_label.config(text=f"✓ Salvo | PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f}")
                except Exception as e:
                    self.status_label.config(text=f"✓ Salvo (erro nas métricas)")
            else:
                self.status_label.config(text=f"✓ Salvo: {out_path.name}")

            # Histórico
            elapsed = time.time() - self.execution_start_time
            self._save_to_history(key, elapsed, True, str(out_path))

            messagebox.showinfo("Sucesso", f"Panorama salvo:\n{out_path}")

        except Exception as e:
            msg = str(e)
            self.status_label.config(text=f"✗ Erro: {msg[:50]}")
            messagebox.showerror("Erro", msg)

            elapsed = time.time() - self.execution_start_time
            label = self.method_var.get()
            key = self.label_to_key.get(label, "unknown")
            self._save_to_history(key, elapsed, False, "")
        finally:
            self.running = False
            self.execution_start_time = None

    def run_all_methods(self):
        """Executa TODOS os métodos e compara"""
        if len(self.files) < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 imagens")
            return

        self._toggle_running(True)
        t = threading.Thread(target=self._run_all_methods_worker, daemon=True)
        t.start()

    def _run_all_methods_worker(self):
        """Worker para executar todos os métodos"""
        try:
            import pandas as pd

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_folder = self.output_dir / f"all_{ts}"
            all_folder.mkdir(parents=True, exist_ok=True)

            self._log(f"Executando todos os métodos... Salvando em: {all_folder.name}")

            has_reference = bool(self.ref_path.get())
            if has_reference:
                self._log(f"Imagem de referência: {Path(self.ref_path.get()).name}")
                self._log("Métricas serão calculadas automaticamente para todos os métodos!")
            else:
                self._log("AVISO: Sem imagem de referência. Métricas PSNR/SSIM não serão calculadas.")

            imgs = read_images(self.files)

            # Ordenar métodos
            classic_keys = sorted([k for k in METHODS.keys() if k.startswith("classic/")])
            deep_keys = sorted([k for k in METHODS.keys() if k.startswith("deep/")])
            ordered_keys = classic_keys + deep_keys

            total = len(ordered_keys)
            success_count = 0
            error_count = 0
            resultados = []

            for idx, key in enumerate(ordered_keys, 1):
                progress = int((idx / total) * 100)
                self.after(0, lambda p=progress: self.progress.config(value=p))
                self.after(0, lambda p=progress: self.progress_label.config(text=f"{p}%"))

                method_label = METHODS[key]["label"]
                self._log(f"[{idx}/{total}] ({progress}%) Executando: {method_label}")

                tempo_inicio = time.time()
                resultado = {
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
                    # Executar com timeout
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        fn = METHODS[key]["fn"]
                        kwargs = {}
                        if METHODS[key].get("accepts_try_gpu", False):
                            kwargs["try_gpu"] = self.try_gpu.get()

                        if key == "classic/opencv_stitcher":
                            future = executor.submit(fn, imgs, modo=self.modo.get(), try_gpu=self.try_gpu.get())
                        else:
                            future = executor.submit(fn, imgs, **kwargs) if kwargs else executor.submit(fn, imgs)

                        pano = future.result(timeout=120)

                    tempo_total = time.time() - tempo_inicio
                    resultado['tempo_segundos'] = tempo_total

                    # Salvar
                    safe_key = key.replace("/", "-")
                    out_path = all_folder / f"pano_{safe_key}.jpg"
                    cv2.imwrite(str(out_path), pano)
                    resultado['arquivo'] = out_path.name
                    resultado['status'] = 'sucesso'

                    # Métricas
                    if has_reference:
                        try:
                            metrics_data = compute_metrics(self.ref_path.get(), pano)

                            # Extrair todas as métricas
                            resultado['psnr_db'] = metrics_data.get('psnr')
                            resultado['ssim'] = metrics_data.get('ssim')
                            resultado['mse'] = metrics_data.get('mse')
                            resultado['ghosting'] = metrics_data.get('ghosting')
                            resultado['seam_energy'] = metrics_data.get('seam_energy')
                            resultado['overlap_ratio'] = metrics_data.get('overlap_ratio')
                            resultado['deltaE_mean'] = metrics_data.get('deltaE_mean')
                            resultado['deltaL_mean'] = metrics_data.get('deltaL_mean')

                            # Calcular métricas adicionais (entropia e nitidez)
                            gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
                            hist, _ = np.histogram(gray, bins=256, range=(0, 256))
                            hist = hist / hist.sum()
                            hist = hist[hist > 0]
                            entropia = -np.sum(hist * np.log2(hist))
                            resultado['entropia'] = entropia

                            # Nitidez (variância Laplaciano)
                            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                            resultado['nitidez'] = laplacian.var()

                            # Log com métricas principais
                            ghosting_val = resultado['ghosting']
                            ghosting_str = f" | Ghost={ghosting_val:.3f}" if ghosting_val is not None and not np.isnan(ghosting_val) else ""
                            seam_val = resultado['seam_energy']
                            seam_str = f" | Seam={seam_val:.2f}" if seam_val is not None and not np.isnan(seam_val) else ""

                            self._log(f"  ✓ Concluído: {out_path.name} | PSNR={metrics_data.get('psnr', 0):.2f} dB | SSIM={metrics_data.get('ssim', 0):.4f}{ghosting_str}{seam_str}")
                        except Exception as met_err:
                            self._log(f"  ✓ Concluído: {out_path.name} (erro nas métricas: {str(met_err)[:50]})")
                    else:
                        self._log(f"  ✓ Concluído: {out_path.name}")

                    success_count += 1

                except Exception as e:
                    error_msg = str(e).split('\n')[0][:100]
                    self._log(f"  ✗ Erro: {error_msg}")
                    resultado['erro'] = error_msg
                    error_count += 1

                resultados.append(resultado)

            # Gerar CSV
            if resultados:
                csv_path = all_folder / "comparacao_metricas.csv"
                df = pd.DataFrame(resultados)

                # Ordenar colunas
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

                # Resumo estatístico
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

            # Abrir pasta automaticamente
            try:
                import os, subprocess, platform
                folder = str(all_folder)
                if platform.system() == "Windows":
                    os.startfile(folder)
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
            except Exception:
                pass

            self.after(0, lambda: messagebox.showinfo("Executar Todos",
                f"Concluído!\n\nSucessos: {success_count}/{total}\nErros: {error_count}/{total}\n\nResultados em: {all_folder.name}"))

        except Exception as e:
            self._log("Erro fatal: " + str(e))
            self.after(0, lambda m=str(e): messagebox.showerror("Erro ao executar todos", m))
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

                import re

                # Detectar padrão das imagens
                has_numeric_pattern = False  # 01.jpg, 02.jpg, 03.jpg
                has_img_pattern = False      # img1.jpg, img(1).jpg

                for img in all_images:
                    stem = img.stem.lower()

                    # Detectar imagem de referência
                    if "reference" in stem:
                        reference_img = img
                        continue

                    # Detectar padrão numérico simples: 01, 02, 03, etc
                    if re.match(r'^\d+$', stem):
                        has_numeric_pattern = True
                        input_images.append(img)
                    # Detectar padrão img(n) ou imgn: img1, img(1), img001, etc
                    elif re.search(r'img\s*\(?\d+\)?', stem):
                        has_img_pattern = True
                        input_images.append(img)

                # Validar combinação de padrões
                if has_numeric_pattern and not reference_img:
                    self._log(f"  ⚠️  ERRO: Padrão numérico (01, 02, 03...) detectado mas sem 'reference.jpg'")
                    self._log(f"  Pulando esta pasta (padrão numérico REQUER referência)...")
                    continue

                if has_img_pattern and reference_img:
                    self._log(f"  ⚠️  AVISO: Padrão img(n) com referência encontrada")
                    self._log(f"  Ignorando referência (padrão img(n) normalmente não usa referência)")
                    reference_img = None

                # Log do padrão detectado
                if has_numeric_pattern:
                    self._log(f"  📋 Padrão detectado: Numérico (01, 02, 03...) + reference.jpg")
                    self._log(f"  ✓ Imagem de referência: {reference_img.name}")
                elif has_img_pattern:
                    self._log(f"  📋 Padrão detectado: img(n) sem referência")
                    self._log(f"  ⚠️  Processando SEM cálculo de métricas")
                else:
                    self._log(f"  ⚠️  Padrão não reconhecido nesta pasta")
                    if reference_img:
                        self._log(f"  ✓ Mas encontrou referência: {reference_img.name}")

                if len(input_images) < 2:
                    self._log(f"  ⚠️  AVISO: Menos de 2 imagens de entrada em {subfolder.name}")
                    self._log(f"  Pulando esta pasta...")
                    continue

                self._log(f"  Imagens de entrada: {len(input_images)}")

                # Decidir estratégia de stitching
                num_imgs = len(input_images)
                use_hierarchical = num_imgs > 5

                if use_hierarchical:
                    self._log(f"  📊 Estratégia: Stitching Hierárquico (>5 imagens)")
                    self._log(f"     - Dividir em grupos de 5")
                    self._log(f"     - Fazer panoramas parciais")
                    self._log(f"     - Juntar panoramas parciais")
                else:
                    self._log(f"  📊 Estratégia: Stitching Direto (≤5 imagens)")

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

                    # Determinar padrão usado
                    if has_numeric_pattern:
                        padrao_usado = 'numerico'
                    elif has_img_pattern:
                        padrao_usado = 'img(n)'
                    else:
                        padrao_usado = 'outro'

                    resultado = {
                        'pasta': subfolder.name,
                        'num_imagens': len(input_images),
                        'padrao': padrao_usado,
                        'tem_referencia': 'sim' if reference_img else 'nao',
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

                        # Decidir se usa stitching hierárquico
                        if use_hierarchical:
                            self._log(f"      → Fase 1: Criando {(num_imgs + 4) // 5} panoramas parciais...")

                            # Dividir em grupos de 5
                            partial_panos = []
                            for i in range(0, num_imgs, 5):
                                group = imgs[i:i+5]
                                group_num = (i // 5) + 1
                                self._log(f"         Grupo {group_num}: {len(group)} imagens")

                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    fn = METHODS[key]["fn"]
                                    kwargs = {}
                                    if METHODS.get(key, {}).get("accepts_try_gpu", False):
                                        kwargs["try_gpu"] = self.try_gpu.get()

                                    if key == "classic/opencv_stitcher":
                                        future = executor.submit(fn, group, modo=self.modo.get(), try_gpu=self.try_gpu.get())
                                    else:
                                        future = executor.submit(fn, group, **kwargs) if kwargs else executor.submit(fn, group)

                                    try:
                                        partial_pano = future.result(timeout=timeout)
                                        partial_panos.append(partial_pano)
                                        self._log(f"         ✓ Grupo {group_num} concluído")
                                    except Exception as e:
                                        raise RuntimeError(f"Erro no grupo {group_num}: {str(e)}")

                            # Fase 2: Juntar panoramas parciais
                            self._log(f"      → Fase 2: Juntando {len(partial_panos)} panoramas parciais...")

                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                fn = METHODS[key]["fn"]
                                kwargs = {}
                                if METHODS.get(key, {}).get("accepts_try_gpu", False):
                                    kwargs["try_gpu"] = self.try_gpu.get()

                                if key == "classic/opencv_stitcher":
                                    future = executor.submit(fn, partial_panos, modo=self.modo.get(), try_gpu=self.try_gpu.get())
                                else:
                                    future = executor.submit(fn, partial_panos, **kwargs) if kwargs else executor.submit(fn, partial_panos)

                                try:
                                    pano = future.result(timeout=timeout)
                                    self._log(f"      ✓ Panorama final criado")
                                except concurrent.futures.TimeoutError:
                                    raise RuntimeError(f"Timeout na fase final ({timeout}s excedido)")
                        else:
                            # Stitching direto (≤5 imagens)
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

                        # Calcular métricas APENAS se houver imagem de referência
                        if reference_img:
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
                        else:
                            # Sem referência, apenas confirma salvamento
                            self._log(f"    ✓ Salvo: {out_filename}")

                        # Gerar montagem before/after se opção estiver ativada E houver referência
                        if self.create_montage.get() and reference_img:
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
                colunas_ordem = ['pasta', 'num_imagens', 'padrao', 'tem_referencia', 'metodo', 'label', 'status', 'tempo_segundos',
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
                    tem_ref = df_pasta['tem_referencia'].iloc[0] == 'sim' if len(df_pasta) > 0 else False
                    padrao = df_pasta['padrao'].iloc[0] if len(df_pasta) > 0 and 'padrao' in df_pasta.columns else 'desconhecido'

                    if len(df_sucesso) > 0:
                        self._log(f"\n  {pasta}:")
                        self._log(f"    Padrão: {padrao}")
                        self._log(f"    Referência: {'✓ Sim' if tem_ref else '✗ Não (sem métricas)'}")
                        self._log(f"    Sucessos: {len(df_sucesso)}/{len(df_pasta)}")

                        if tem_ref:
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
                        os.startfile(folder)
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

        self._log(f"\n{'='*60}")
        self._log("📊 GERANDO RELATÓRIO TOP 5 MÉTODOS")
        self._log(f"{'='*60}")

        # Filtrar apenas sucessos e pastas COM referência
        df_sucesso = df[(df['status'] == 'sucesso') & (df['tem_referencia'] == 'sim')].copy()

        if len(df_sucesso) == 0:
            self._log("⚠️ Nenhum método teve sucesso em pastas com referência.")
            self._log("   Top 5 só pode ser gerado para pastas que têm imagem de referência.")
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

        # Normalizar PSNR (maior é melhor)
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
                pasta = result_row.get('pasta', '')
                arquivo = result_row['arquivo']

                if arquivo:
                    # Copiar imagem resultado
                    src_img = scan_folder / arquivo
                    if src_img.exists():
                        dst_img = top5_folder / f"rank{idx:02d}_{arquivo}"
                        shutil.copy2(src_img, dst_img)

                    # Copiar montagem before/after se existir
                    if self.create_montage.get():
                        montage_name = arquivo.replace('.jpg', '_comparison.jpg')
                        src_montage = scan_folder / "before_after" / montage_name
                        if src_montage.exists():
                            dst_montage = top5_folder / f"rank{idx:02d}_{montage_name}"
                            shutil.copy2(src_montage, dst_montage)

        # Gerar relatório detalhado em TXT (VERSÃO RESUMIDA)
        report_path = top5_folder / "RELATORIO_TOP5_METODOS.txt"
        self._log(f"\n📄 Gerando relatório detalhado: {report_path.name}")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RELATÓRIO: TOP 5 MÉTODOS DE IMAGE STITCHING\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de métodos avaliados: {len(df_metodos)}\n\n")

            f.write("=" * 80 + "\n")
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

                f.write(f"\nTempo médio: {row['tempo_segundos']:.2f}s | Sucessos: {int(row['num_sucessos'])}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("=" * 80 + "\n")

        self._log(f"✓ Relatório gerado com sucesso!")
        self._log(f"✓ {len(top5)} métodos copiados para: {top5_folder.name}")
        self._log(f"{'='*60}\n")

    # ===== Advanced Stitching =====
    def run_directional(self):
        """Executa stitching direcional"""
        if len(self.files) < 2:
            messagebox.showwarning("Aviso", "Selecione pelo menos 2 imagens")
            return

        direction = self.direction_var.get()
        self.advanced_status.config(text=f"Executando {direction}...")

        def task():
            try:
                images = [cv2.imread(str(f)) for f in self.files]
                stitcher = DirectionalStitcher(direction=direction)
                result, info = stitcher.stitch(images)

                if result is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output = self.output_dir / f"panorama_{direction}_{ts}.jpg"
                    cv2.imwrite(str(output), result)
                    self.advanced_status.config(text=f"✓ Salvo: {output.name}")
                    messagebox.showinfo("Sucesso", f"Salvo:\n{output}")
                else:
                    self.advanced_status.config(text="✗ Erro")
                    messagebox.showerror("Erro", str(info))
            except Exception as e:
                self.advanced_status.config(text=f"✗ Erro: {str(e)[:50]}")
                messagebox.showerror("Erro", str(e))

        threading.Thread(target=task, daemon=True).start()

    def run_grid(self):
        """Executa grid stitching"""
        grid_type = self.grid_var.get()
        required = {"2x2": 4, "3x3": 9, "2x3": 6}.get(grid_type, 0)

        if grid_type != "auto" and len(self.files) != required:
            messagebox.showwarning("Aviso", f"Grid {grid_type} requer {required} imagens. Você tem {len(self.files)}.")
            return

        self.advanced_status.config(text=f"Executando grid {grid_type}...")

        def task():
            try:
                images = [cv2.imread(str(f)) for f in self.files]

                if grid_type == "auto":
                    stitcher = GridStitcher()
                    result, info = stitcher.stitch_flexible_grid(images, auto_arrange=True)
                    grid_used = info.get('grid_size', 'auto')
                else:
                    rows, cols = map(int, grid_type.split('x'))
                    stitcher = GridStitcher(grid_size=(rows, cols))
                    result, info = stitcher.stitch_grid(images)
                    grid_used = grid_type

                if result is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output = self.output_dir / f"panorama_grid_{grid_used}_{ts}.jpg"
                    cv2.imwrite(str(output), result)
                    self.advanced_status.config(text=f"✓ Grid {grid_used} salvo")
                    messagebox.showinfo("Sucesso", f"Salvo:\n{output}")
                else:
                    self.advanced_status.config(text="✗ Erro")
                    messagebox.showerror("Erro", str(info))
            except Exception as e:
                self.advanced_status.config(text=f"✗ Erro: {str(e)[:50]}")
                messagebox.showerror("Erro", str(e))

        threading.Thread(target=task, daemon=True).start()

    # ===== YOLO Methods =====
    def yolo_select_folder(self):
        """Seleciona pasta para YOLO"""
        folder = filedialog.askdirectory(title="Pasta para YOLO")
        if not folder:
            return
        folder_path = Path(folder)
        count = 0
        for ext in SUPPORTED_EXTS:
            for img in folder_path.glob(f"*{ext}"):
                if img not in self.yolo_files:
                    self.yolo_files.append(img)
                    self.yolo_listbox.insert(tk.END, img.name)
                    count += 1
        self.yolo_img_count.config(text=f"{len(self.yolo_files)} imagens")
        self.yolo_status.config(text=f"✓ {count} imagens adicionadas")

    def yolo_add_files(self):
        """Adiciona arquivos para YOLO"""
        paths = filedialog.askopenfilenames(
            title="Imagens para YOLO",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp"), ("Todos", "*.*")]
        )
        count = 0
        for p in paths:
            path = Path(p)
            if path not in self.yolo_files:
                self.yolo_files.append(path)
                self.yolo_listbox.insert(tk.END, path.name)
                count += 1
        self.yolo_img_count.config(text=f"{len(self.yolo_files)} imagens")
        if count > 0:
            self.yolo_status.config(text=f"✓ {count} imagens adicionadas")

    def yolo_clear_files(self):
        """Limpa lista YOLO"""
        self.yolo_files.clear()
        self.yolo_listbox.delete(0, tk.END)
        self.yolo_img_count.config(text="0 imagens")
        self.yolo_status.config(text="Lista limpa")

    def load_yolo_model(self):
        """Carrega modelo YOLO"""
        self.yolo_status.config(text="Carregando modelo...")

        def task():
            try:
                version = self.yolo_version.get()
                size = self.yolo_size.get()
                device = self.yolo_device.get()

                self.yolo_detector = YOLODetector(
                    model_version=version,
                    model_size=size,
                    device=device
                )
                self.yolo_status.config(text=f"✓ Modelo {version}{size} carregado ({device})")
                messagebox.showinfo("Sucesso", f"Modelo {version}{size} carregado!")
            except Exception as e:
                self.yolo_status.config(text="✗ Erro ao carregar")
                messagebox.showerror("Erro", f"Falha:\n{e}")

        threading.Thread(target=task, daemon=True).start()

    def run_yolo_detection(self):
        """Executa detecção YOLO"""
        if self.yolo_detector is None:
            messagebox.showwarning("Aviso", "Carregue o modelo primeiro")
            return
        if len(self.yolo_files) == 0:
            messagebox.showwarning("Aviso", "Selecione imagens")
            return

        total_images = len(self.yolo_files)
        self.yolo_status.config(text=f"Detectando em {total_images} imagens...")

        def task():
            try:
                detect_mode = self.detect_what.get()
                conf = self.yolo_conf.get()
                total_detections = 0

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                detection_dir = self.output_dir / f"yolo_detection_{ts}"
                detection_dir.mkdir(exist_ok=True)

                for idx, img_path in enumerate(self.yolo_files, 1):
                    self.yolo_status.config(text=f"Processando {idx}/{total_images}: {img_path.name}")

                    image = cv2.imread(str(img_path))

                    # Detectar
                    if detect_mode == "cars":
                        detections = self.yolo_detector.detect_cars(image, conf)
                    elif detect_mode == "persons":
                        detections = self.yolo_detector.detect_persons(image, conf)
                    elif detect_mode == "both":
                        detections = self.yolo_detector.detect_cars_and_persons(image, conf)
                    else:
                        detections = self.yolo_detector.detect(image, conf)

                    total_detections += detections['total_detections']

                    # Visualizar e salvar
                    result = self.yolo_detector.visualize(image, detections)
                    output = detection_dir / f"detection_{img_path.stem}.jpg"
                    cv2.imwrite(str(output), result)

                    # JSON
                    json_output = detection_dir / f"detection_{img_path.stem}.json"
                    self.yolo_detector.save_results(str(json_output), detections, str(img_path))

                self.yolo_status.config(text=f"✓ {total_detections} objetos detectados em {total_images} imagens")
                messagebox.showinfo("Sucesso",
                    f"Detecção concluída!\n\n"
                    f"Imagens: {total_images}\n"
                    f"Objetos: {total_detections}\n\n"
                    f"Resultados em:\n{detection_dir}")

            except Exception as e:
                self.yolo_status.config(text=f"✗ Erro: {str(e)[:50]}")
                messagebox.showerror("Erro", str(e))

        threading.Thread(target=task, daemon=True).start()

    def run_yolo_on_panorama(self):
        """Detecta no último panorama"""
        panoramas = sorted(self.output_dir.glob("panorama_*.jpg"))
        if not panoramas:
            messagebox.showwarning("Aviso", "Nenhum panorama encontrado")
            return

        if self.yolo_detector is None:
            messagebox.showwarning("Aviso", "Carregue o modelo YOLO primeiro")
            return

        latest = panoramas[-1]
        self.yolo_status.config(text="Detectando em panorama...")

        def task():
            try:
                image = cv2.imread(str(latest))
                detect_mode = self.detect_what.get()
                conf = self.yolo_conf.get()

                if detect_mode == "cars":
                    detections = self.yolo_detector.detect_cars(image, conf)
                elif detect_mode == "persons":
                    detections = self.yolo_detector.detect_persons(image, conf)
                elif detect_mode == "both":
                    detections = self.yolo_detector.detect_cars_and_persons(image, conf)
                else:
                    detections = self.yolo_detector.detect(image, conf)

                result = self.yolo_detector.visualize(image, detections)
                output = self.output_dir / f"detection_panorama_{latest.stem}.jpg"
                cv2.imwrite(str(output), result)

                self.yolo_status.config(text=f"✓ {detections['total_detections']} objetos detectados")
                messagebox.showinfo("Resultado",
                    f"Detectados {detections['total_detections']} objetos:\n"
                    f"{detections['count_by_class']}\n\n"
                    f"Salvo: {output.name}")

            except Exception as e:
                self.yolo_status.config(text=f"✗ Erro: {str(e)[:50]}")
                messagebox.showerror("Erro", str(e))

        threading.Thread(target=task, daemon=True).start()

    # ===== Video Methods =====
    def select_video(self):
        """Seleciona vídeo"""
        path = filedialog.askopenfilename(
            title="Selecionar vídeo",
            filetypes=[("Vídeos", "*.mp4 *.avi *.mov *.mkv"), ("Todos", "*.*")]
        )
        if path:
            self.video_path.set(path)

    def show_video_info(self):
        """Mostra info do vídeo"""
        if not self.video_path.get():
            messagebox.showwarning("Aviso", "Selecione um vídeo")
            return

        try:
            processor = VideoProcessor(self.video_path.get())
            info = (f"Vídeo: {Path(self.video_path.get()).name}\n\n"
                   f"Resolução: {processor.width}x{processor.height}\n"
                   f"FPS: {processor.fps:.2f}\n"
                   f"Total de frames: {processor.total_frames}\n"
                   f"Duração: {processor.duration:.2f}s")
            messagebox.showinfo("Info do Vídeo", info)
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def run_video_processing(self):
        """Processa vídeo"""
        if not self.video_path.get():
            messagebox.showwarning("Aviso", "Selecione um vídeo")
            return

        self.video_status.config(text="Processando vídeo...")

        def task():
            try:
                processor = VideoProcessor(self.video_path.get())
                method = self.extract_method.get()
                interval = self.video_interval.get()
                max_frames = self.video_max_frames.get()

                # Extrair
                if method == "interval":
                    frames = processor.extract_frames_by_interval(interval, max_frames)
                elif method == "time":
                    frames = processor.extract_frames_by_time(interval, max_frames)
                else:
                    frames = processor.extract_key_frames(max_frames=max_frames)

                # Salvar
                video_name = Path(self.video_path.get()).stem
                frames_dir = self.output_dir / f"video_{video_name}_frames"
                frames_dir.mkdir(exist_ok=True)

                for i, frame in enumerate(frames):
                    cv2.imwrite(str(frames_dir / f"frame_{i:04d}.jpg"), frame)

                # Ação adicional
                action = self.video_action.get()

                if action == "stitch" and len(frames) >= 2:
                    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
                    status, pano = stitcher.stitch(frames)
                    if status == ST_OK:
                        cv2.imwrite(str(self.output_dir / f"video_{video_name}_panorama.jpg"), pano)

                elif action == "grid" and len(frames) >= 4:
                    grid_stitcher = GridStitcher()
                    result, _ = grid_stitcher.stitch_flexible_grid(frames, auto_arrange=True)
                    if result is not None:
                        cv2.imwrite(str(self.output_dir / f"video_{video_name}_grid.jpg"), result)

                elif action == "detect" and self.yolo_detector:
                    for i, frame in enumerate(frames):
                        detections = self.yolo_detector.detect(frame)
                        result = self.yolo_detector.visualize(frame, detections)
                        cv2.imwrite(str(frames_dir / f"detection_{i:04d}.jpg"), result)

                self.video_status.config(text=f"✓ {len(frames)} frames processados")
                messagebox.showinfo("Sucesso",
                    f"Vídeo processado!\n"
                    f"Frames: {len(frames)}\n"
                    f"Salvos em: {frames_dir}")

            except Exception as e:
                self.video_status.config(text=f"✗ Erro: {str(e)[:50]}")
                messagebox.showerror("Erro", str(e))

        threading.Thread(target=task, daemon=True).start()

    # ===== Utility Methods =====
    def _log(self, msg: str):
        """Escreve mensagem no console de log"""
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.configure(state=tk.DISABLED)
        self.log.see(tk.END)

    def _toggle_running(self, running: bool):
        """Gerencia estado dos botões durante execução"""
        def set_state():
            state = tk.DISABLED if running else tk.NORMAL
            self.run_btn.config(state=state)
            self.run_all_btn.config(state=state)
            self.scan_folder_btn.config(state=state)
            if not running:
                self.progress.config(value=0)
                self.progress_label.config(text="0%")
        self.after(0, set_state)

    def _update_timer(self):
        """Atualiza timer"""
        if self.execution_start_time:
            elapsed = int(time.time() - self.execution_start_time)
            mins = elapsed // 60
            secs = elapsed % 60
            self.timer_label.config(text=f"{mins:02d}:{secs:02d}")
            self.after(1000, self._update_timer)

    def _load_history(self):
        """Carrega histórico"""
        if self.history_file.exists():
            try:
                return json.loads(self.history_file.read_text(encoding="utf-8"))
            except:
                return []
        return []

    def _save_to_history(self, method_key, elapsed, success, output_path):
        """Salva no histórico"""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_key,
            "elapsed": elapsed,
            "success": success,
            "output": str(output_path),
            "num_images": len(self.files)
        }
        self.history.append(entry)
        self.history_file.write_text(json.dumps(self.history, indent=2), encoding="utf-8")

    def _load_presets(self):
        """Carrega presets"""
        if self.presets_file.exists():
            try:
                return json.loads(self.presets_file.read_text(encoding="utf-8"))
            except:
                return {}
        return {}

    def save_preset(self):
        """Salva preset"""
        name = tk.simpledialog.askstring("Salvar Preset", "Nome do preset:")
        if not name:
            return

        preset = {
            "method": self.method_var.get(),
            "modo": self.modo.get(),
            "try_gpu": self.try_gpu.get(),
            "calc_metrics": self.calc_metrics.get(),
            "ref_path": self.ref_path.get(),
            "create_montage": self.create_montage.get(),
        }
        self.presets[name] = preset
        self.presets_file.write_text(json.dumps(self.presets, indent=2), encoding="utf-8")
        messagebox.showinfo("Sucesso", f"Preset '{name}' salvo!")

    def load_preset(self):
        """Carrega preset"""
        if not self.presets:
            messagebox.showinfo("Info", "Nenhum preset salvo")
            return

        name = tk.simpledialog.askstring("Carregar Preset", f"Presets disponíveis:\n{', '.join(self.presets.keys())}\n\nNome:")
        if not name or name not in self.presets:
            return

        preset = self.presets[name]
        self.method_var.set(preset.get("method", ""))
        self.modo.set(preset.get("modo", "panorama"))
        self.try_gpu.set(preset.get("try_gpu", False))
        self.calc_metrics.set(preset.get("calc_metrics", False))
        self.ref_path.set(preset.get("ref_path", ""))
        self.create_montage.set(preset.get("create_montage", False))

        messagebox.showinfo("Sucesso", f"Preset '{name}' carregado!")

    def show_history(self):
        """Mostra histórico"""
        if not self.history:
            messagebox.showinfo("Histórico", "Nenhum histórico disponível")
            return

        win = tk.Toplevel(self)
        win.title("Histórico de Execução")
        win.geometry("800x400")

        tree = ttk.Treeview(win, columns=("timestamp", "method", "time", "status"), show="headings")
        tree.heading("timestamp", text="Data/Hora")
        tree.heading("method", text="Método")
        tree.heading("time", text="Tempo (s)")
        tree.heading("status", text="Status")

        for entry in reversed(self.history[-50:]):  # Últimas 50
            tree.insert("", tk.END, values=(
                entry["timestamp"],
                entry["method"],
                f"{entry['elapsed']:.1f}",
                "✓ Sucesso" if entry["success"] else "✗ Erro"
            ))

        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Button(win, text="Fechar", command=win.destroy).pack(pady=10)

    def show_help(self):
        """Mostra ajuda"""
        help_text = """
ATALHOS DE TECLADO:

Ctrl+O       Adicionar imagens
Ctrl+R       Executar stitching
F5           Executar todos os métodos
Delete       Remover selecionado
Ctrl+L       Limpar lista
Ctrl+S       Salvar preset
Ctrl+P       Carregar preset
Ctrl+H       Mostrar histórico
F1           Esta ajuda

RECURSOS PRINCIPAIS:

🔧 Stitching Completo: 20+ métodos (clássicos + deep learning)
🎯 Grid & Direcional: Arranjos especiais
🚗 Detecção YOLO: Detecção de objetos
🎬 Vídeo: Processamento de vídeos
⚡ Executar Todos: Compara todos os métodos
📊 Métricas: PSNR, SSIM, ghosting, seam energy
💾 Presets: Salvar/carregar configurações
"""
        messagebox.showinfo("Ajuda", help_text)


def main():
    """Executa a aplicação"""
    app = EnhancedStitcherGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
