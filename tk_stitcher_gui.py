# -*- coding: utf-8 -*-
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from typing import List
import json

# ===== Dependências =====
try:
    import cv2
except Exception as err:
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("Erro", "OpenCV não encontrado. Instale com: pip install opencv-python\nDetalhes: " + str(err))
    raise

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

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

# ===== Registry de métodos =====
METHODS = {}

def register_method(key, label):
    def deco(fn):
        METHODS[key] = {"label": label, "fn": fn, "accepts_try_gpu": True}
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
        # >>> crie a StringVar ANTES de construir a UI da direita
        self.out_dir_var = tk.StringVar(value=str(self.output_dir))

        self._build_left()
        self._build_right()

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

        sort_bar = ttk.Frame(left)
        sort_bar.pack(fill=tk.X, pady=6)
        ttk.Button(sort_bar, text="▲ Subir", command=lambda: self.move_selection(-1)).pack(side=tk.LEFT)
        ttk.Button(sort_bar, text="▼ Descer", command=lambda: self.move_selection(1)).pack(side=tk.LEFT, padx=6)
        ttk.Button(sort_bar, text="Remover selecionados", command=self.remove_selected).pack(side=tk.LEFT)

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

        mode_row = ttk.Frame(box)
        mode_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mode_row, text="Modo:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Panorama", value="panorama", variable=self.modo).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(mode_row, text="Scans", value="scans", variable=self.modo).pack(side=tk.LEFT, padx=(8, 0))

        gpu_row = ttk.Frame(box)
        gpu_row.pack(fill=tk.X, pady=6)
        ttk.Checkbutton(gpu_row, text="Tentar GPU (Deep/LoFTR)", variable=self.try_gpu).pack(side=tk.LEFT)

        met_box = ttk.LabelFrame(right, text="Métricas (opcional)")
        met_box.pack(fill=tk.X, pady=(10, 0))
        ttk.Checkbutton(met_box, text="Calcular métricas vs referência (PSNR/SSIM)", variable=self.do_metrics).pack(anchor=tk.W)
        ref_row = ttk.Frame(met_box)
        ref_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(ref_row, text="Referência:").pack(side=tk.LEFT)
        ttk.Entry(ref_row, textvariable=self.ref_path, width=44).pack(side=tk.LEFT, padx=6)
        ttk.Button(ref_row, text="Escolher...", command=self.choose_ref).pack(side=tk.LEFT)

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
        self.run_btn = ttk.Button(act_box, text="Executar Stitch", command=self.run_stitch)
        self.run_btn.pack(side=tk.LEFT, padx=(0, 6))
        self.run_all_btn = ttk.Button(act_box, text="Executar Todos", command=self.run_all_methods)
        self.run_all_btn.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(act_box, mode="indeterminate", length=200)
        self.progress.pack(side=tk.LEFT, padx=10)

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

    def clear_files(self):
        self.files.clear()
        self.listbox.delete(0, tk.END)
        self._log("Lista limpa.")

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            self.listbox.delete(idx)
            del self.files[idx]
        self._log(f"Removidos {len(sel)} item(ns).")

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
            if self.do_metrics.get() and self.ref_path.get():
                try:
                    metrics_data = compute_metrics(ref_path=self.ref_path.get(), test_img=pano)
                    calc_done = True
                except Exception as met_err:
                    self._log("Falha nas métricas: " + str(met_err))

            save_metrics_txt(
                metrics_txt,
                key,
                ts,
                metrics_data,
                out_path.name,
                self.ref_path.get() or "(sem referência)"
            )
            append_metrics_log(self.output_dir, safe_key, ts, metrics_data, extra={"out": out_path.name})

            if calc_done:
                self._log(f"Métricas: PSNR={metrics_data.get('psnr', float('nan')):.3f} dB | SSIM={metrics_data.get('ssim', float('nan')):.4f}")

            self._log("Concluído: " + str(out_path))
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
            self.after(0, lambda m=msg: messagebox.showerror("Erro no stitch", m))
        finally:
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

        try:
            # Criar pasta com timestamp
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            all_folder = self.output_dir / f"all_{ts}"
            all_folder.mkdir(parents=True, exist_ok=True)

            self._log(f"Executando todos os métodos... Salvando em: {all_folder.name}")

            imgs = read_images(self.files)

            # Ordenar métodos: classics primeiro, depois deep
            all_keys = list(METHODS.keys())
            classic_keys = sorted([k for k in all_keys if k.startswith("classic/")])
            deep_keys = sorted([k for k in all_keys if k.startswith("deep/")])
            ordered_keys = classic_keys + deep_keys

            total = len(ordered_keys)
            success_count = 0
            error_count = 0

            for idx, key in enumerate(ordered_keys, 1):
                method_label = METHODS[key]["label"]
                self._log(f"[{idx}/{total}] Executando: {method_label}")

                try:
                    fn = METHODS[key]["fn"]

                    # Passa try_gpu apenas se o método realmente aceitar
                    kwargs = {}
                    if METHODS.get(key, {}).get("accepts_try_gpu", False):
                        kwargs["try_gpu"] = self.try_gpu.get()

                    if key == "classic/opencv_stitcher":
                        pano = fn(imgs, modo=self.modo.get(), try_gpu=self.try_gpu.get())
                    else:
                        pano = fn(imgs, **kwargs) if kwargs else fn(imgs)

                    # Salvar resultado
                    safe_key = key.replace("/", "-")
                    out_path = all_folder / f"pano_{safe_key}.jpg"
                    ok = cv2.imwrite(str(out_path), pano)
                    if not ok:
                        raise RuntimeError("Falha ao salvar a imagem de saída.")

                    self._log(f"  ✓ Concluído: {out_path.name}")
                    success_count += 1

                except Exception as e:
                    error_msg = str(e).split('\n')[0][:100]  # Primeira linha, até 100 chars
                    self._log(f"  ✗ Erro: {error_msg}")
                    error_count += 1

            # Resumo final
            self._log(f"\n{'='*50}")
            self._log(f"Concluído! Sucessos: {success_count}/{total}, Erros: {error_count}/{total}")
            self._log(f"Resultados salvos em: {all_folder}")
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

            self.after(0, lambda: messagebox.showinfo("Executar Todos",
                f"Concluído!\n\nSucessos: {success_count}/{total}\nErros: {error_count}/{total}\n\nResultados em: {all_folder.name}"))

        except Exception as exc:
            msg = str(exc)
            self._log("Erro fatal: " + msg)
            self.after(0, lambda m=msg: messagebox.showerror("Erro ao executar todos", m))
        finally:
            self._toggle_running(False)

    # ---------- Utils ----------
    def _toggle_running(self, running: bool):
        def set_state():
            self.run_btn.config(state=(tk.DISABLED if running else tk.NORMAL))
            self.run_all_btn.config(state=(tk.DISABLED if running else tk.NORMAL))
            if running:
                self.progress.start(10)
            else:
                self.progress.stop()
        self.after(0, set_state)

    def _log(self, msg: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.configure(state=tk.DISABLED)
        self.log.see(tk.END)

    def _notify_done(self, out_path: Path):
        self.after(0, lambda: messagebox.showinfo("Concluído", "Panorama salvo em: " + str(out_path)))

# ---------- Métricas ----------
def compute_metrics(ref_path: str, test_img):
    import numpy as np, cv2
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
    return {"psnr": psnr, "ssim": ssim}

if __name__ == "__main__":
    app = StitcherGUI()
    app.mainloop()
