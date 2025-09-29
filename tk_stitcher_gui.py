# -*- coding: utf-8 -*-
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from typing import List

# ===== Dependências =====
try:
    import cv2
except Exception as err:
    # Evita erro de janela antes de criar root
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("Erro", "OpenCV não encontrado. Instale com: pip install opencv-python Detalhes: " + str(err))
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
    _ = try_gpu  # placeholder
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


# ===== Registry de métodos =====
METHODS = {}


def register_method(key, label):
    def deco(fn):
        METHODS[key] = {"label": label, "fn": fn}
        return fn
    return deco


@register_method("classic/opencv_stitcher", "Clássico • OpenCV Stitcher (panorama/scans)")
def method_opencv_stitcher(imgs, modo="panorama", try_gpu=False):
    st = make_stitcher(modo, try_gpu)
    status, pano = st.stitch(imgs)
    if status != ST_OK:
        raise RuntimeError(status_to_msg(status))
    return pano


@register_method("classic/orb_homography", "Clássico • ORB + Homography (sequencial)")
def method_orb_homography(imgs, ratio=0.75, ransac_reproj=4.0):
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
    """Chama seu modules/deep/loftr.py como módulo Python em subprocesso.
    Expectativa: o script aceita --dir <pasta> e --out <arquivo> e salva o panorama.
    Ajuste se sua CLI for diferente.
    """
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
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            raise RuntimeError("LoFTR falhou (retcode " + str(ret.returncode) + ") STDOUT: " + ret.stdout + " STDERR: " + ret.stderr)
        pano = _cv.imread(str(out_file))
        if pano is None:
            raise RuntimeError("LoFTR concluiu mas não gerou imagem em: " + str(out_file))
        return pano
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



@register_method("deep/superglue", "Deep • SuperGlue (placeholder)")
def method_superglue_placeholder(imgs, **kwargs):
    raise RuntimeError("SuperGlue não está configurado. Integre modules/deep/sp.superglue.py e exponha stitch(imgs).")


@register_method("deep/flow_stitch", "Deep • Optical Flow Stitch (placeholder)")
def method_flow_placeholder(imgs, **kwargs):
    raise RuntimeError("Flow Stitch não está configurado. Integre modules/deep/flow_stitch.py e exponha stitch(imgs).")


class StitcherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stitcher — Clássico & Deep")
        self.geometry("980x600")
        self.minsize(900, 560)
        self.configure(padx=10, pady=10)

        self.files: List[Path] = []
        self.output_dir = Path.cwd() / "results"
        self.modo = tk.StringVar(value="panorama")
        self.try_gpu = tk.BooleanVar(value=False)
        self.method_key = tk.StringVar(value=list(METHODS.keys())[0])

        self._build_left()
        self._build_right()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir_var.set(str(self.output_dir))

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

        # Método
        meth_row = ttk.Frame(box)
        meth_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(meth_row, text="Método:").pack(side=tk.LEFT)
        values = list(METHODS.keys())
        self.method_cb = ttk.Combobox(meth_row, textvariable=self.method_key, state="readonly", width=50, values=values)
        self.method_cb.pack(side=tk.LEFT, padx=8)

        # Modo
        mode_row = ttk.Frame(box)
        mode_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mode_row, text="Modo:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Panorama", value="panorama", variable=self.modo).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(mode_row, text="Scans", value="scans", variable=self.modo).pack(side=tk.LEFT, padx=(8, 0))

        gpu_row = ttk.Frame(box)
        gpu_row.pack(fill=tk.X, pady=6)
        ttk.Checkbutton(gpu_row, text="Tentar GPU (futuro)", variable=self.try_gpu).pack(side=tk.LEFT)

        out_box = ttk.LabelFrame(right, text="Saída")
        out_box.pack(fill=tk.X, pady=(10, 0))
        self.out_dir_var = tk.StringVar()
        out_row = ttk.Frame(out_box)
        out_row.pack(fill=tk.X, pady=6)
        ttk.Label(out_row, text="Pasta:").pack(side=tk.LEFT)
        self.out_entry = ttk.Entry(out_row, textvariable=self.out_dir_var, width=44)
        self.out_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(out_row, text="Escolher...", command=self.choose_output_dir).pack(side=tk.LEFT)

        act_box = ttk.Frame(right)
        act_box.pack(fill=tk.X, pady=(12, 0))
        self.run_btn = ttk.Button(act_box, text="Executar Stitch", command=self.run_stitch)
        self.run_btn.pack(side=tk.LEFT)
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
        self._log("Adicionados " + str(add) + " arquivo(s). Total: " + str(len(self.files)))

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
        self._log("Removidos " + str(len(sel)) + " item(ns).")

    def move_selection(self, delta: int):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for i in (sel if delta > 0 else reversed(sel)):
            j = i + delta
            if 0 <= j < self.listbox.size():
                self.files[i], self.files[j] = self.files[j], self.files[i]
                txt_i = self.listbox.get(i)
                txt_j = self.listbox.get(j)
                self.listbox.delete(i)
                self.listbox.insert(i, txt_j)
                self.listbox.delete(j)
                self.listbox.insert(j, txt_i)
        self.listbox.selection_clear(0, tk.END)
        for i in [k + delta for k in sel if 0 <= k + delta < self.listbox.size()]:
            self.listbox.selection_set(i)

    def choose_output_dir(self):
        d = filedialog.askdirectory(title="Escolha a pasta de saída", initialdir=str(self.output_dir))
        if d:
            self.output_dir = Path(d)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.out_dir_var.set(str(self.output_dir))

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
            key = self.method_key.get()
            fn = METHODS[key]["fn"]
            if key == "classic/opencv_stitcher":
                pano = fn(imgs, modo=self.modo.get(), try_gpu=self.try_gpu.get())
            else:
                pano = fn(imgs)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_key = key.replace("/", "-")
            out_path = self.output_dir / ("pano_" + safe_key + "_" + ts + ".jpg")
            ok = cv2.imwrite(str(out_path), pano)
            if not ok:
                raise RuntimeError("Falha ao salvar a imagem de saída.")
            self._log("Concluído: " + str(out_path))
            self._notify_done(out_path)
        except Exception as exc:
            msg = str(exc)
            self._log("Erro: " + msg)
            self.after(0, lambda m=msg: messagebox.showerror("Erro no stitch", m))
        finally:
            self._toggle_running(False)

    # ---------- Utils ----------
    def _toggle_running(self, running: bool):
        def set_state():
            self.run_btn.config(state=(tk.DISABLED if running else tk.NORMAL))
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


if __name__ == "__main__":
    app = StitcherGUI()
    app.mainloop()
