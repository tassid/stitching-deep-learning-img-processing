# Novos Módulos - Funcionalidades Avançadas

Este documento descreve os novos módulos implementados para expandir as capacidades do projeto **sem impactar o código existente**.

## 📁 Estrutura

```
modules/
├── detection/          # Detecção de objetos com YOLO
│   ├── __init__.py
│   └── yolo_detector.py
├── advanced_stitching/ # Stitching direcional e em grid
│   ├── __init__.py
│   ├── directional_stitcher.py
│   └── grid_stitcher.py
└── video/             # Processamento de vídeo
    ├── __init__.py
    └── video_processor.py
```

---

## 🎯 Módulo 1: Detection (YOLO)

### Funcionalidades
- Detecção de objetos usando YOLOv5, YOLOv8 ou YOLOv11
- Detecção específica de **carros** e **pessoas**
- Comparação entre detecção em imagens individuais vs panoramas
- Visualização de resultados
- Salvamento de resultados em JSON

### Uso Básico

```python
from modules.detection import YOLODetector

# Inicializar detector
detector = YOLODetector(
    model_version='yolov8',  # ou 'yolov5', 'yolov11'
    model_size='n',          # n, s, m, l, x
    device='cuda'            # ou 'cpu'
)

# Detectar objetos
detections = detector.detect(image, conf_threshold=0.25)

# Detectar apenas carros
cars = detector.detect_cars(image)

# Detectar apenas pessoas
persons = detector.detect_persons(image)

# Detectar carros E pessoas
both = detector.detect_cars_and_persons(image)

# Visualizar resultados
result_img = detector.visualize(image, detections)
cv2.imwrite("resultado.jpg", result_img)

# Comparar detecções: individuais vs panorama
comparison = detector.compare_detections(
    individual_images=[img1, img2, img3],
    panorama=panorama_img
)
print(f"Melhoria: {comparison['analysis']['improvement_percentage']:.2f}%")
```

### Classes Suportadas (COCO Dataset)
- `0`: person
- `2`: car
- Outras classes disponíveis (consulte COCO_CLASSES no código)

---

## 🔄 Módulo 2: Advanced Stitching

### 2.1 Directional Stitcher

#### Funcionalidades
- Stitching **horizontal** (esquerda → direita)
- Stitching **vertical** (cima → baixo)
- Compatível com qualquer método de stitching base

#### Uso

```python
from modules.advanced_stitching import DirectionalStitcher

# Stitching horizontal
stitcher_h = DirectionalStitcher(direction='horizontal')
panorama_h, info = stitcher_h.stitch(images)

# Stitching vertical
stitcher_v = DirectionalStitcher(direction='vertical')
panorama_v, info = stitcher_v.stitch(images)

# Usar método customizado como base
# from modules.classic.metodo_x import stitch
# stitcher = DirectionalStitcher(direction='horizontal', base_method=stitch)
```

### 2.2 Grid Stitcher

#### Funcionalidades
- Stitching em **grid 2x2** (4 imagens)
- Stitching em **grid 3x3** (9 imagens)
- Grids customizados (NxM)
- Auto-arranjo flexível

#### Uso

```python
from modules.advanced_stitching import GridStitcher

# Grid 2x2
stitcher = GridStitcher(grid_size=(2, 2))
panorama_2x2, info = stitcher.stitch_2x2(images_4)

# Grid 3x3
panorama_3x3, info = stitcher.stitch_3x3(images_9)

# Grid customizado 2x4
panorama_custom, info = stitcher.stitch_custom_grid(images_8, rows=2, cols=4)

# Auto-arranjo (detecta melhor grid)
panorama_auto, info = stitcher.stitch_flexible_grid(images, auto_arrange=True)
```

#### Arranjo Automático
- 4 imagens → 2x2
- 6 imagens → 2x3
- 9 imagens → 3x3
- Outros → grid aproximadamente quadrado

---

## 🎬 Módulo 3: Video Processing

### Funcionalidades
- Extração de frames por **intervalo** (ex: a cada 10 frames)
- Extração por **tempo** (ex: 1 frame por segundo)
- Extração de **keyframes** (frames com mudança significativa)
- Salvamento automático em disco
- Processamento com callback customizado

### Uso Básico

```python
from modules.video import VideoProcessor, extract_frames

# Carregar vídeo
processor = VideoProcessor("video.mp4")

# Método 1: Extrair a cada 10 frames
frames = processor.extract_frames_by_interval(
    interval=10,      # a cada 10 frames
    max_frames=20     # máximo 20 frames
)

# Método 2: Extrair por tempo (1 frame/segundo)
frames = processor.extract_frames_by_time(
    time_interval=1.0,  # segundos
    max_frames=30
)

# Método 3: Extrair keyframes (mudanças significativas)
keyframes = processor.extract_key_frames(
    threshold=30.0,      # limiar de diferença
    min_interval=10,     # mínimo 10 frames entre keyframes
    max_frames=15
)

# Salvar frames em disco
saved_paths = processor.extract_and_save_frames(
    output_dir="frames_output",
    interval=10,
    prefix="frame",
    format="jpg"
)

# Obter frame específico
frame_100 = processor.get_frame_at(100)
frame_at_5s = processor.get_frame_at_time(5.0)
```

### Função Auxiliar Rápida

```python
# Atalho simples
frames = extract_frames(
    video_path="video.mp4",
    interval=10,
    method='interval'  # ou 'time', 'keyframes'
)
```

---

## 🔗 Workflow Integrado

### Exemplo: Vídeo → Stitching → Detecção

```python
from modules.video import VideoProcessor
from modules.advanced_stitching import GridStitcher
from modules.detection import YOLODetector

# 1. Extrair frames do vídeo
processor = VideoProcessor("video.mp4")
frames = processor.extract_frames_by_interval(interval=10, max_frames=4)

# 2. Criar panorama 2x2
stitcher = GridStitcher(grid_size=(2, 2))
panorama, info = stitcher.stitch_grid(frames)

# 3. Detectar objetos
detector = YOLODetector(model_version='yolov8', model_size='n')
detections = detector.detect_cars_and_persons(panorama)

# 4. Visualizar
result = detector.visualize(panorama, detections)
cv2.imwrite("resultado_final.jpg", result)

print(f"Detectados: {detections['total_detections']} objetos")
print(f"Por classe: {detections['count_by_class']}")
```

---

## 📊 Novos Datasets Suportados

### Datasets Configurados

1. **APAP Dataset - Case 17**
   ```python
   path = r"C:\datasets\APAP Dataset (2013)\images2\case17"
   ```

2. **APAP Dataset - Case 26**
   ```python
   path = r"C:\datasets\APAP Dataset (2013)\images2\case26"
   ```

3. **Deep Learning Stitching - DeepH Data**
   ```python
   path = r"C:\datasets\Deep Learning Image Stitching\Content-Aware-DeepH-Data (2020)\Contant-Aware-DeepH-Data\Data\Test"
   ```

### Usar com Novos Datasets

```python
from pathlib import Path

dataset_path = Path(r"C:\datasets\APAP Dataset (2013)\images2\case17")
images = list(dataset_path.glob("*.jpg"))

# Carregar imagens
imgs = [cv2.imread(str(img)) for img in images[:4]]

# Processar
stitcher = GridStitcher(grid_size=(2, 2))
panorama, info = stitcher.stitch_grid(imgs)
```

---

## 🚀 Executar Demo

Um script de demonstração completo está disponível:

```bash
python demo_new_features.py
```

**Nota:** O script vem com exemplos comentados. Descomente as linhas e ajuste os caminhos para seus arquivos para executar.

---

## 📝 Exemplos de Casos de Uso

### Caso 1: Panorama com Pessoas
```python
# Dataset com várias pessoas
images = [cv2.imread(f"pessoas/img{i}.jpg") for i in range(4)]

# Criar panorama 2x2
stitcher = GridStitcher(grid_size=(2, 2))
panorama, _ = stitcher.stitch_grid(images)

# Detectar pessoas
detector = YOLODetector()
detections = detector.detect_persons(panorama)

print(f"Pessoas detectadas: {detections['count_by_class']['person']}")
```

### Caso 2: Vídeo de Trânsito
```python
# Extrair frames de vídeo de trânsito
processor = VideoProcessor("trafego.mp4")
frames = processor.extract_frames_by_interval(interval=30)

# Detectar carros em cada frame
detector = YOLODetector()
for i, frame in enumerate(frames):
    cars = detector.detect_cars(frame)
    print(f"Frame {i}: {cars['count_by_class'].get('car', 0)} carros")
```

### Caso 3: Imagens Verticais
```python
# Stitching vertical (fotos de prédios, por exemplo)
stitcher = DirectionalStitcher(direction='vertical')
panorama, _ = stitcher.stitch(vertical_images)
```

---

## ⚙️ Requisitos

```bash
# YOLO Detection
pip install ultralytics  # YOLOv8/v11
pip install torch torchvision  # YOLOv5

# Stitching (já instalado no projeto)
opencv-python>=4.5.0

# Vídeo (já incluso no OpenCV)
```

---

## 🔧 Integração com Código Existente

**Importante:** Estes módulos são **completamente independentes** e não afetam o código existente.

Você pode:
- ✅ Usar junto com métodos clássicos existentes
- ✅ Usar junto com métodos deep existentes
- ✅ Combinar com as métricas existentes
- ✅ Manter o pipeline original intacto

### Exemplo de Integração

```python
# Usar método clássico existente + novo grid stitcher
from modules.classic.sift_stitcher import stitch_with_sift
from modules.advanced_stitching import GridStitcher

# Grid com método SIFT como base
stitcher = GridStitcher(base_method=stitch_with_sift)
panorama, _ = stitcher.stitch_2x2(images)
```

---

## 📖 Próximos Passos

1. **Testar com seus dados:**
   - Descomente exemplos no `demo_new_features.py`
   - Ajuste caminhos para seus arquivos
   - Execute e analise resultados

2. **Integrar com TCC:**
   - Adicionar seção sobre detecção de objetos
   - Comparar detecção em individuais vs panoramas
   - Análise de vídeos

3. **Experimentos sugeridos:**
   - Testar diferentes grids (2x2, 3x3, 2x4)
   - Comparar horizontal vs vertical
   - Analisar impacto de ghosting na detecção

---

## 🐛 Troubleshooting

### Erro: "Model not found"
```bash
# Instalar YOLOv8/v11
pip install ultralytics

# Ou YOLOv5
pip install torch torchvision
```

### Erro: "Video cannot be opened"
- Verifique se o codec está instalado
- Tente converter vídeo: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

### Stitching falha em grid
- Verifique sobreposição entre imagens adjacentes
- Tente com `auto_arrange=True` em `stitch_flexible_grid`

---

## 📧 Suporte

Para dúvidas ou problemas:
1. Verifique o `demo_new_features.py`
2. Consulte documentação inline nos arquivos
3. Abra uma issue no repositório

---

**Desenvolvido para o TCC - Sistema de Fusão de Imagens e Detecção de Objetos**
