import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime
from typing import List

# Dependências: OpenCV (cv2). Pillow é opcional (apenas se quiser pré-visualização).
try:
    import cv2
except ImportError:
    messagebox.showerror("Erro", "Instale o OpenCV: pip install opencv-python")
    raise

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def make_stitcher(modo: str, try_gpu: bool):
    """Cria e configura o stitcher clássico do OpenCV (PANORAMA ou SCANS)."""
    mode = cv2.Stitcher_PANORAMA if modo == "panorama" else cv2.Stitcher_SCANS
    st = cv2.Stitcher_create(mode)
    # Confiança padrão um pouco mais permissiva
    st.setPanoConfidenceThresh(0.5)
    # Warper esférico quando disponível (costuma estabilizar panoramas)
    try:
        Warper = getattr(cv2, "PyRotationWarper", None)
        if Warper and hasattr(st, "setWarper"):
            st.setWarper(Warper("spherical", 1.0))
    except Exception:
        pass
    # OpenCV Python não expõe GPU toggle direto aqui; mantido para compatibilidade futura
    _ = try_gpu
    return st


def read_images(files: List[Path]):
    imgs = []
    for p in files:
        im = cv2.imread(str(p))
        if im is None:
            raise RuntimeError(f"Falha ao ler: {p}")
        imgs.append(im)
    if len(imgs) < 2:
        raise RuntimeError("Selecione pelo menos 2 imagens.")
    return imgs


class StitcherGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stitcher GUI")
        self.geometry("860x560")
        self.minsize(820, 520)
        self.configure(padx=10, pady=10)

        # Estado
        self.files: List[Path] = []
        self.output_dir = Path.cwd() / "results"
        self.modo = tk.StringVar(value="panorama")
        self.try_gpu = tk.BooleanVar(value=False)

        # Layout principal: esquerda (lista) / direita (controles + log)
        self._build_left()
        self._build_right()

        # Define diretório inicial
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir_var.set(str(self.output_dir))

    # UI
    def _build_left(self):
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Barra de botões (adicionar/limpar)
        bar = ttk.Frame(left)
        bar.pack(fill=tk.X)
        ttk.Button(bar, text="Adicionar imagens…", command=self.add_files).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Adicionar pasta…", command=self.add_folder).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(bar, text="Limpar lista", command=self.clear_files).pack(side=tk.LEFT)

        # Lista + scrollbar
        self.listbox = tk.Listbox(left, selectmode=tk.EXTENDED)
        yscroll = ttk.Scrollbar(left, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=yscroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(8, 0))
        yscroll.pack(side=tk.LEFT, fill=tk.Y, pady=(8, 0))

        # Controles de ordenação
        sort_bar = ttk.Frame(left)
        sort_bar.pack(fill=tk.X, pady=6)
        ttk.Button(sort_bar, text="▲ Subir", command=lambda: self.move_selection(-1)).pack(side=tk.LEFT)
        ttk.Button(sort_bar, text="▼ Descer", command=lambda: self.move_selection(1)).pack(side=tk.LEFT, padx=6)
        ttk.Button(sort_bar, text="Remover selecionados", command=self.remove_selected).pack(side=tk.LEFT)

    def _build_right(self):
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(10, 0))

        # Parâmetros
        box = ttk.LabelFrame(right, text="Parâmetros")
        box.pack(fill=tk.X)

        mode_row = ttk.Frame(box)
        mode_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(mode_row, text="Modo:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Panorama", value="panorama", variable=self.modo).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Radiobutton(mode_row, text="Scans", value="scans", variable=self.modo).pack(side=tk.LEFT, padx=(8, 0))

        gpu_row = ttk.Frame(box)
        gpu_row.pack(fill=tk.X, pady=6)
        ttk.Checkbutton(gpu_row, text="Tentar GPU (futuro)", variable=self.try_gpu).pack(side=tk.LEFT)

        # Saída
        out_box = ttk.LabelFrame(right, text="Saída")
        out_box.pack(fill=tk.X, pady=(10, 0))

        self.out_dir_var = tk.StringVar()
        out_row = ttk.Frame(out_box)
        out_row.pack(fill=tk.X, pady=6)
        ttk.Label(out_row, text="Pasta:").pack(side=tk.LEFT)
        self.out_entry = ttk.Entry(out_row, textvariable=self.out_dir_var, width=44)
        self.out_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(out_row, text="Escolher…", command=self.choose_output_dir).pack(side=tk.LEFT)

        # Ações
        act_box = ttk.Frame(right)
        act_box.pack(fill=tk.X, pady=(12, 0))
        self.run_btn = ttk.Button(act_box, text="Executar Stitch", command=self.run_stitch)
        self.run_btn.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(act_box, mode="indeterminate", length=200)
        self.progress.pack(side=tk.LEFT, padx=10)

        # Log
        log_box = ttk.LabelFrame(right, text="Log")
        log_box.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
        self.log = tk.Text(log_box, height=18, width=40, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True)

    # Handlers
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
        # Evita duplicados; mantém ordem de chegada
        existing = set(map(str, self.files))
        for p in new_files:
            if str(p) not in existing:
                self.files.append(p)
                self.listbox.insert(tk.END, str(p))
        self._log(f"Adicionados {len(new_files)} arquivo(s). Total: {len(self.files)}")

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
                # swap na lista e no listbox
                self.files[i], self.files[j] = self.files[j], self.files[i]
                txt_i = self.listbox.get(i)
                txt_j = self.listbox.get(j)
                self.listbox.delete(i)
                self.listbox.insert(i, txt_j)
                self.listbox.delete(j)
                self.listbox.insert(j, txt_i)
        # re-seleciona novos índices
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
        # Thread para não travar a UI
        t = threading.Thread(target=self._run_stitch_worker, daemon=True)
        self._toggle_running(True)
        t.start()

    def _run_stitch_worker(self):
        try:
            imgs = read_images(self.files)
            st = make_stitcher(self.modo.get(), self.try_gpu.get())
            self._log(f"Rodando stitch em {len(imgs)} imagem(ns) | modo={self.modo.get()}…")
            status, pano = st.stitch(imgs)
            if status != cv2.STITCHER_OK:
                raise RuntimeError(self._status_to_msg(status))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = self.output_dir / f"pano_{self.modo.get()}_{ts}.jpg"
            ok = cv2.imwrite(str(out_path), pano)
            if not ok:
                raise RuntimeError("Falha ao salvar a imagem de saída.")
            self._log(f"✅ Concluído: {out_path}")
            self._notify_done(out_path)
        except Exception as e:
            msg = str(e)
            self._log(f"❌ Erro: {msg}")
            # Use default-arg binding to avoid late-binding NameError when Tk calls the lambda later
            self.after(0, lambda m=msg: messagebox.showerror("Erro no stitch", m))
        finally:
            self._toggle_running(False)

    # Utils
    def _toggle_running(self, running: bool):
        def set_state():
            self.run_btn.config(state=(tk.DISABLED if running else tk.NORMAL))
            if running:
                self.progress.start(10)
            else:
                self.progress.stop()
        self.after(0, set_state)

    def _status_to_msg(self, status: int) -> str:
        mapping = {
            cv2.STITCHER_ERR_NEED_MORE_IMGS: "Não há imagens suficientes (>=2) ou sobreposição insuficiente.",
            cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL: "Falha ao estimar homografia (poucas features ou não correspondem).",
            cv2.STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Falha ao ajustar parâmetros da câmera.",
        }
        # Status 1/"ERR_NEED_MORE_IMGS" também aparece quando não há sobreposição suficiente
        base = mapping.get(status, f"Falhou com status {status}.")
        tip = "\nDicas: verifique ordem/sequência, garanta sobreposição (~30–50%), iluminações parecidas, e use 'scans' para documentos/varreduras."
        return base + tip

    def _log(self, msg: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.configure(state=tk.DISABLED)
        self.log.see(tk.END)

    def _notify_done(self, out_path: Path):
        self.after(0, lambda: messagebox.showinfo("Concluído", f"Panorama salvo em:\n{out_path}"))


if __name__ == "__main__":
    # macOS: se der erro de _tkinter, instale Python com Tk (ex.: via brew: brew install python-tk) ou use o instalador oficial.
    app = StitcherGUI()
    app.mainloop()
