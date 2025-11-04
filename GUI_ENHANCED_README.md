# GUI Enhanced - Interface Completa

Interface gráfica expandida com todos os novos recursos integrados.

## 🚀 Executar

```bash
python tk_stitcher_gui_enhanced.py
```

## 📋 Abas Disponíveis

### 1. 📐 Stitching Clássico
- Interface simplificada para stitching tradicional
- Adicionar imagens individuais ou pastas
- Executar stitching panorâmico
- Abrir pasta de resultados

### 2. 🎯 Grid & Direcional

#### Stitching Direcional
- **Horizontal** (→): Esquerda para direita
- **Vertical** (↓): Cima para baixo

#### Grid Stitching
- **2x2**: 4 imagens em grade
- **3x3**: 9 imagens em grade
- **2x3**: 6 imagens em grade
- **Auto**: Arranjo automático baseado no número de imagens

### 3. 🚗 Detecção YOLO

#### Configuração do Modelo
- **Versões**: YOLOv5, YOLOv8, YOLOv11
- **Tamanhos**: n (nano), s (small), m (medium), l (large), x (xlarge)
- **Dispositivo**: CPU ou CUDA (GPU)

#### Opções de Detecção
- 🚗 **Apenas Carros**
- 👤 **Apenas Pessoas**
- 🚗👤 **Carros E Pessoas**
- 🌐 **Todos os Objetos** (COCO dataset completo)

#### Funcionalidades
- Detectar em imagens individuais
- Detectar em panoramas gerados
- Ajustar confiança mínima (slider)
- Modo comparação: individuais vs panorama

### 4. 🎬 Vídeo

#### Métodos de Extração
- **Por intervalo**: A cada N frames (ex: 10)
- **Por tempo**: A cada N segundos (ex: 1.0)
- **Keyframes**: Apenas frames com mudanças significativas

#### Parâmetros
- Intervalo configurável
- Máximo de frames para extrair
- Info do vídeo (resolução, FPS, duração)

#### Ações Pós-Extração
- 💾 **Apenas salvar frames**
- 🎯 **Salvar + Fazer stitching**
- 🚗 **Salvar + Detecção YOLO**
- ⊞ **Salvar + Grid stitching**

## 🎨 Características da Interface

### Design Moderno
- Interface com abas (Notebook)
- Ícones descritivos (emojis)
- Status bars específicos por aba
- Barra de progresso global
- Tooltips informativos

### Facilidades
- Threading para não travar interface
- Mensagens de erro/sucesso claras
- Salvamento automático na pasta `results/`
- Preview de imagens (na aba clássico)
- Histórico de operações

## 📊 Resultados

Todos os resultados são salvos em `results/` com nomenclatura descritiva:

```
results/
├── panorama_classic_20250103_143052.jpg
├── panorama_horizontal_20250103_143120.jpg
├── panorama_grid_2x2_20250103_143145.jpg
├── detection_image1.jpg
├── detection_panorama_classic_20250103_143052.jpg
└── video_traffico_frames/
    ├── frame_0000.jpg
    ├── frame_0001.jpg
    └── detection_0000.jpg
```

## 🔧 Requisitos

```bash
# Base (obrigatório)
pip install opencv-python pillow

# Para YOLO (opcional)
pip install ultralytics  # YOLOv8/v11
pip install torch torchvision  # YOLOv5

# Para GPU (opcional, mas recomendado)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Como Usar

### Stitching Básico
1. Vá para aba **"Stitching Clássico"**
2. Clique **"Adicionar Imagens"** ou **"Adicionar Pasta"**
3. Clique **"Executar Stitching"**
4. Resultado salvo automaticamente

### Grid 2x2
1. Vá para aba **"Grid & Direcional"**
2. Adicione **4 imagens** na lista (aba Clássico)
3. Selecione **"2x2 (4 imagens)"**
4. Clique **"Executar Grid"**

### Detecção em Panorama
1. Gere um panorama primeiro (qualquer método)
2. Vá para aba **"Detecção YOLO"**
3. Clique **"Carregar Modelo"**
4. Escolha o que detectar (carros, pessoas, etc)
5. Clique **"Detectar em Panorama"**

### Processar Vídeo
1. Vá para aba **"Vídeo"**
2. Clique **"Selecionar"** e escolha um vídeo
3. Configure extração (ex: a cada 10 frames)
4. Escolha ação (ex: Salvar + Grid stitching)
5. Clique **"Processar Vídeo"**

## 🎯 Workflows Recomendados

### Workflow 1: Panorama com Detecção
```
1. [Clássico] Adicionar imagens
2. [Clássico] Executar stitching
3. [YOLO] Carregar modelo
4. [YOLO] Detectar em panorama
```

### Workflow 2: Vídeo → Grid → Detecção
```
1. [Vídeo] Selecionar vídeo
2. [Vídeo] Configurar: a cada 10 frames, máx 4
3. [Vídeo] Ação: Salvar + Grid stitching
4. [YOLO] Carregar modelo
5. [YOLO] Detectar em panorama gerado
```

### Workflow 3: Grid Manual
```
1. [Clássico] Adicionar 9 imagens
2. [Grid & Direcional] Selecionar "3x3"
3. [Grid & Direcional] Executar Grid
```

## ⚙️ Configurações Avançadas

### Otimização de Performance

**Para melhor qualidade:**
- YOLO: modelo "l" ou "x"
- Grid: usar imagens de alta resolução
- Vídeo: extrair mais frames

**Para melhor velocidade:**
- YOLO: modelo "n" ou "s"
- Grid: redimensionar imagens antes
- Vídeo: extrair menos frames

### GPU vs CPU

**Recomendações:**
- **YOLO**: Sempre use GPU se disponível (10-50x mais rápido)
- **Stitching**: GPU tem impacto menor
- **Vídeo**: CPU geralmente suficiente para extração

## 🐛 Troubleshooting

### YOLO não carrega
```bash
# Instalar dependências
pip install ultralytics torch torchvision
```

### Stitching falha
- Verifique sobreposição entre imagens (30-50%)
- Tente modo "Scans" ao invés de "Panorama"
- Para vertical, tente girar imagens 90°

### Grid falha
- Certifique-se de ter o número exato de imagens
- Use "Auto" se número não for exato
- Verifique que imagens têm tamanho similar

### Vídeo não abre
```bash
# Instalar codecs
pip install opencv-python-headless
```

## 📝 Atalhos de Teclado

*Nota: Atalhos podem ser adicionados em versões futuras*

## 🔄 Diferenças da GUI Original

| Recurso | GUI Original | GUI Enhanced |
|---------|-------------|--------------|
| Stitching clássico | ✅ Completo | ✅ Simplificado |
| Métodos deep | ✅ 20+ métodos | ⚠️ Básico |
| Métricas | ✅ PSNR/SSIM | ❌ Não incluído |
| Detecção YOLO | ❌ | ✅ Completo |
| Grid stitching | ❌ | ✅ 2x2, 3x3, auto |
| Direcional | ❌ | ✅ H/V |
| Vídeo | ❌ | ✅ Completo |
| Interface | 🪟 Single window | 📑 Tabs |

## 🚀 Executar Ambas as GUIs

```bash
# GUI original (completa para stitching)
python tk_stitcher_gui.py

# GUI enhanced (novos recursos)
python tk_stitcher_gui_enhanced.py
```

**Recomendação:** Use ambas conforme necessidade:
- **Original**: Testes comparativos de métodos, métricas detalhadas
- **Enhanced**: Detecção, vídeos, grids, workflows completos

## 📦 Estrutura de Código

```python
EnhancedStitcherGUI
├── __init__()
├── _build_notebook()          # Cria abas
├── _build_classic_tab()       # Aba stitching
├── _build_advanced_tab()      # Aba grid/direcional
├── _build_yolo_tab()          # Aba YOLO
├── _build_video_tab()         # Aba vídeo
├── _build_status_bar()        # Barra status
├── run_classic_stitch()       # Executa stitching
├── run_directional()          # Executa direcional
├── run_grid()                 # Executa grid
├── load_yolo_model()          # Carrega YOLO
├── run_yolo_detection()       # Detecta objetos
├── run_video_processing()     # Processa vídeo
└── open_results_folder()      # Abre results/
```

## 🎓 Exemplos de Uso

### Exemplo 1: Panorama Simples
```
1. Adicionar 3 imagens
2. Executar stitching
3. Ver resultado em results/
```

### Exemplo 2: Detectar Carros em Panorama
```
1. Gerar panorama (3+ imagens de rua)
2. Carregar YOLOv8n
3. Selecionar "Apenas Carros"
4. Detectar em panorama
5. Ver detecções salvas
```

### Exemplo 3: Grid de Vídeo
```
1. Selecionar vídeo de drone
2. Extrair 9 frames (a cada 30)
3. Executar grid 3x3
4. Ver panorama gerado
```

## 🔗 Integração com Código Existente

A GUI enhanced **não substitui** a original, mas complementa:

```python
# Usar módulos diretamente
from modules.detection import YOLODetector
from modules.advanced_stitching import GridStitcher

# Ou usar GUI
from tk_stitcher_gui_enhanced import EnhancedStitcherGUI
app = EnhancedStitcherGUI()
app.mainloop()
```

## 📧 Suporte

Para problemas:
1. Verifique requisitos instalados
2. Consulte NOVOS_MODULOS.md
3. Execute demo_new_features.py para testar módulos
4. Abra issue no repositório

---

**Desenvolvido para o TCC - Sistema de Fusão de Imagens e Detecção de Objetos**
