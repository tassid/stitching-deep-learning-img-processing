# Melhorias Práticas para a UI

## ✨ Melhorias Rápidas e Impactantes:

### 1. **Status Bar Informativo** (5 min)
Adicione uma barra de status no rodapé mostrando:
- Número de imagens carregadas: `📁 2 imagens`
- Método selecionado: `🔧 SIFT + Homografia`
- Pasta de saída: `💾 results/`
- Status: `✓ Pronto` / `⏳ Processando...`

```python
# No final do __init__, adicione:
self.status_bar = ttk.Label(self, text="Pronto", relief=tk.SUNKEN, anchor=tk.W)
self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

# Atualize em add_files():
self.status_bar.config(text=f"📁 {len(self.files)} imagens carregadas")
```

### 2. **Contador Visual de Imagens** (3 min)
Mostre quantas imagens estão na lista:

```python
# Após criar self.listbox, adicione um label:
self.img_count_label = ttk.Label(left, text="0 imagens", font=("", 9, "italic"))
self.img_count_label.pack(anchor=tk.W, pady=(4, 0))

# Atualize em add_files() e clear_files():
def _update_count(self):
    count = len(self.files)
    self.img_count_label.config(text=f"{count} {'imagem' if count == 1 else 'imagens'}")
```

### 3. **Preview da Primeira Imagem** (10 min)
Mostre um thumbnail da primeira imagem selecionada:

```python
# No _build_left(), após sort_bar:
preview_frame = ttk.LabelFrame(left, text="Preview")
preview_frame.pack(fill=tk.X, pady=6)
self.preview_label = ttk.Label(preview_frame, text="Selecione imagens...")
self.preview_label.pack(pady=10)

# Adicione método:
def _show_preview(self):
    if not self.files:
        self.preview_label.config(image="", text="Selecione imagens...")
        return

    try:
        from PIL import Image, ImageTk
        img = Image.open(self.files[0])
        img.thumbnail((200, 150))
        photo = ImageTk.PhotoImage(img)
        self.preview_label.config(image=photo, text="")
        self.preview_label.image = photo  # Mantém referência
    except:
        self.preview_label.config(text="Erro ao carregar preview")

# Chame em add_files():
self._show_preview()
```

### 4. **Progresso com Porcentagem** (5 min)
Atualize a progressbar para mostrar %:

```python
# Substitua self.bar atual por:
self.bar = ttk.Progressbar(runbox, mode="determinate", maximum=100)

# No run_all(), atualize:
for idx, key in enumerate(ordered_keys):
    progress = int((idx / total) * 100)
    self.bar["value"] = progress
    self.bar.update()
    self._log(f"[{idx+1}/{total}] ({progress}%) Executando: {label}")
```

### 5. **Tooltips Descritivos nos Métodos** (10 min)
Adicione descrições ao passar o mouse:

```python
# Dicionário de descrições:
METHOD_DESCRIPTIONS = {
    "classic/opencv_stitcher": "Stitcher oficial do OpenCV. Rápido e confiável.",
    "classic/orb_homography": "Features ORB + RANSAC. Bom para imagens com pouca textura.",
    "deep/loftr": "Detector-free matching. Excelente para baixa textura.",
    # ... adicione todos os métodos
}

# Após criar self.method_cb:
def on_method_change(event):
    key = self.label_to_key[self.method_label.get()]
    desc = METHOD_DESCRIPTIONS.get(key, "Sem descrição")
    self.status_bar.config(text=f"🔧 {desc}")

self.method_cb.bind("<<ComboboxSelected>>", on_method_change)
```

### 6. **Botão "Re-executar Último"** (5 min)
Repete a última execução com um clique:

```python
# No _build_right(), em runbox, adicione:
ttk.Button(runbox, text="🔄 Re-executar", command=self.rerun_last).pack(...)

# Adicione método:
def rerun_last(self):
    if not self.last_result_path or not self.files:
        messagebox.showwarning("Aviso", "Nenhuma execução anterior encontrada")
        return
    self.run()
```

### 7. **Estimativa de Tempo** (8 min)
Mostre tempo estimado baseado no histórico:

```python
# Adicione no __init__:
self.execution_times = {}  # {method_key: [time1, time2, ...]}

# No final de run():
method_key = self.label_to_key[self.method_label.get()]
elapsed = time.time() - start_time
if method_key not in self.execution_times:
    self.execution_times[method_key] = []
self.execution_times[method_key].append(elapsed)

# Antes de run(), mostre estimativa:
avg_time = sum(self.execution_times[method_key]) / len(self.execution_times[method_key])
self._log(f"⏱️ Tempo estimado: ~{avg_time:.1f}s")
```

### 8. **Tema Dark/Light** (15 min)
Toggle simples de tema:

```python
# No _build_right(), adicione:
theme_row = ttk.Frame(box)
theme_row.pack(fill=tk.X, pady=6)
ttk.Button(theme_row, text="🌙 Dark", command=lambda: self.set_theme("dark")).pack(side=tk.LEFT)
ttk.Button(theme_row, text="☀️ Light", command=lambda: self.set_theme("light")).pack(side=tk.LEFT, padx=4)

def set_theme(self, theme):
    if theme == "dark":
        self.configure(bg="#2b2b2b")
        # Configure outros widgets...
    else:
        self.configure(bg="SystemButtonFace")
```

### 9. **Salvar/Carregar Presets** (20 min)
Salve configurações favoritas:

```python
import json

def save_preset(self):
    name = simpledialog.askstring("Preset", "Nome do preset:")
    if not name:
        return

    preset = {
        "method": self.method_label.get(),
        "mode": self.modo.get(),
        "gpu": self.try_gpu.get()
    }

    presets_file = Path("presets.json")
    presets = {}
    if presets_file.exists():
        presets = json.loads(presets_file.read_text())

    presets[name] = preset
    presets_file.write_text(json.dumps(presets, indent=2))
    messagebox.showinfo("Sucesso", f"Preset '{name}' salvo!")

def load_preset(self):
    # Similar, mas carrega e aplica configurações
    pass
```

### 10. **Atalhos Úteis Adicionais** (3 min)
Já tem atalhos, mas adicione mais:

```python
# Em _setup_shortcuts():
self.bind("<Control-s>", lambda e: self.save_preset())  # Ctrl+S: Salvar preset
self.bind("<Control-l>", lambda e: self.load_preset())  # Ctrl+L: Carregar preset
self.bind("<Delete>", lambda e: self.remove_selected())  # Del: Remover selecionados
self.bind("<Control-q>", lambda e: self.quit())  # Ctrl+Q: Sair
```

## 📊 Impacto Estimado:

| Melhoria | Tempo | Impacto | Prioridade |
|----------|-------|---------|------------|
| Status Bar | 5 min | ⭐⭐⭐⭐⭐ | 🔴 Alta |
| Contador Imagens | 3 min | ⭐⭐⭐⭐ | 🔴 Alta |
| Preview Imagem | 10 min | ⭐⭐⭐⭐⭐ | 🔴 Alta |
| Progresso % | 5 min | ⭐⭐⭐⭐ | 🔴 Alta |
| Tooltips Métodos | 10 min | ⭐⭐⭐⭐⭐ | 🔴 Alta |
| Re-executar | 5 min | ⭐⭐⭐ | 🟡 Média |
| Estimativa Tempo | 8 min | ⭐⭐⭐ | 🟡 Média |
| Tema Dark/Light | 15 min | ⭐⭐ | 🟢 Baixa |
| Presets | 20 min | ⭐⭐⭐ | 🟡 Média |
| Atalhos Extras | 3 min | ⭐⭐ | 🟢 Baixa |

## 🎯 Recomendação:

**Implemente PRIMEIRO** (alta prioridade, 38 minutos):
1. Status Bar (5 min)
2. Contador Imagens (3 min)
3. Preview Imagem (10 min)
4. Progresso % (5 min)
5. Tooltips Métodos (10 min)
6. Re-executar (5 min)

Isso dará um **upgrade visual significativo** com menos de 1 hora de trabalho!
