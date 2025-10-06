# 🖼️ Image Stitcher Pro - GUI

Software profissional de stitching de imagens com 20 métodos diferentes (clássicos e deep learning).

## ✨ Funcionalidades

### 🎯 Core Features
- **20 métodos de stitching**: 15 clássicos + 5 deep learning
- **Execução individual ou batch**: Execute um método ou todos de uma vez
- **Preview em tempo real**: Thumbnail da primeira imagem selecionada
- **Barra de progresso**: Mostra progresso % e tempo decorrido
- **Status bar informativa**: Mostra contador de imagens, método selecionado e descrição

### 💾 Gestão de Configurações
- **Presets**: Salve e carregue configurações favoritas
- **Histórico de execuções**: Rastreamento completo de todas execuções
- **Exportação de relatórios**: Gere relatórios HTML das execuções em batch
- **Persistência**: Configurações salvas automaticamente

### 🎨 Interface
- **Tema Dark/Light**: Alterne entre temas claro e escuro
- **Drag & Drop**: Arraste imagens diretamente para a lista (requer tkinterdnd2)
- **Tooltips descritivos**: Descrições detalhadas de cada método
- **Timer em tempo real**: Acompanhe tempo de execução

### ⌨️ Atalhos de Teclado

#### Operações Básicas
- `Ctrl+O` - Adicionar imagens
- `Ctrl+R` - Executar stitch
- `F5` - Executar todos os métodos
- `Delete` - Remover imagens selecionadas
- `Ctrl+L` - Limpar lista

#### Configurações
- `Ctrl+S` - Salvar preset
- `Ctrl+P` - Carregar preset
- `Ctrl+T` - Alternar tema Dark/Light

#### Análise
- `Ctrl+H` - Ver histórico de execuções
- `Ctrl+E` - Exportar relatório batch
- `F1` - Mostrar ajuda

## 📊 Métodos Disponíveis

### Clássicos (15 métodos)
1. **OpenCV Stitcher** - Stitcher oficial do OpenCV. Rápido e confiável.
2. **ORB + Homography** - Features ORB + RANSAC. Bom para pouca textura.
3. **SIFT Básico** - SIFT + homografia simples. Base sólida e rápida.
4. **SIFT RANSAC Feather** - SIFT manual com feather blending suave.
5. **Features Homography** - Features clássicos com múltiplas opções.
6. **Blend Exposure** - Feather blend baseado em distância. Natural e rápido.
7. **Blend Wavelet** - Fusão por wavelets (Daubechies). Preserva detalhes.
8. **Seams Multiband** - Graph-cut seams + multiband. Profissional.
9. **Harris/FAST** - Detectores de cantos. Leve e rápido.
10. **Blend Laplacian** - Pirâmide Laplaciana multi-escala. Suave.
11. **Exposure Comp** - Compensação de exposição. Remove diferenças de luz.
12. **Blend Poisson** - Seamless cloning de Poisson. Ótimo para costuras.
13. **Blend Multifocal** - Blend baseado em foco. Combina áreas nítidas.
14. **Blend Hybrid** - Contraste, saturação e exposição. Flexível.
15. **ECC Alignment** - Enhanced Correlation Coefficient. Alinhamento preciso.

### Deep Learning (5 métodos)
16. **Deep Homography** - CNN para homografia. Robusto a deslocamentos.
17. **Flow Stitch** - SIFT robusto com fallback. Panoramas gerais.
18. **LoFTR** - Detector-free matching. Excelente para pouca textura.
19. **SuperGlue** - SIFT com blend adaptativo. Robusto e versátil.
20. **UDIS** - SIFT simplificado. Rápido e eficiente.

## 🚀 Como Usar

### Instalação
```bash
# Dependências básicas
pip install opencv-python opencv-contrib-python numpy pillow

# Opcional: Drag & Drop
pip install tkinterdnd2

# Opcional: Deep Learning
pip install torch torchvision kornia
```

### Execução
```bash
python tk_stitcher_gui.py
```

### Fluxo de Trabalho
1. **Adicionar Imagens**: Clique em "Adicionar imagens..." ou arraste arquivos
2. **Ordenar**: Use ▲/▼ para ajustar ordem de stitching
3. **Escolher Método**: Selecione no dropdown (veja descrição na status bar)
4. **Configurar**: Ajuste modo (Panorama/Scans) e GPU se disponível
5. **Executar**: Clique em "▶ Executar" ou pressione `Ctrl+R`
6. **Comparar**: Use "▶▶ Todos" (`F5`) para executar todos os métodos

### Presets
Salve configurações favoritas:
1. Configure método, modo, GPU, métricas
2. Pressione `Ctrl+S` ou clique em "💾 Salvar Preset"
3. Digite um nome descritivo
4. Carregue depois com `Ctrl+P` ou "📂 Carregar Preset"

### Histórico
Acesse o histórico completo:
1. Pressione `Ctrl+H` ou clique em "📊 Histórico"
2. Veja últimas 50 execuções com tempo, status e número de imagens
3. Exporte para CSV para análise externa

### Relatórios Batch
Após executar "Todos":
1. Pressione `Ctrl+E` ou clique em "📄 Relatório"
2. Relatório HTML será gerado com resumo de todos métodos
3. Abre automaticamente no navegador

## 📁 Estrutura de Saída

```
results/
├── pano_classic-opencv_stitcher_20251005_120000.jpg    # Resultado individual
├── pano_classic-opencv_stitcher_20251005_120000.txt    # Métricas (se habilitado)
├── all_20251005_120500/                                 # Batch execution
│   ├── pano_classic-opencv_stitcher.jpg
│   ├── pano_classic-orb_homography.jpg
│   ├── ...
│   └── report.html                                      # Relatório gerado
├── metrics_log.txt                                      # Log consolidado
└── history.json                                         # Histórico de execuções

presets.json                                             # Presets salvos (raiz)
```

## 🎨 Temas

### Light (Padrão)
- Fundo claro, ideal para ambientes bem iluminados
- Contraste alto para leitura fácil

### Dark
- Fundo escuro (#2b2b2b), reduz fadiga ocular
- Ideal para longas sessões ou ambientes escuros
- Ative com `Ctrl+T` ou botão "🌙 Dark"

## 📊 Métricas (Opcional)

Compare resultados com imagem de referência:
1. Marque "Calcular métricas vs referência"
2. Escolha imagem de referência (ground truth)
3. Após execução, veja PSNR e SSIM no log
4. Métricas salvas em arquivo `.txt` junto ao resultado

**PSNR**: Peak Signal-to-Noise Ratio (dB) - quanto maior, melhor
**SSIM**: Structural Similarity Index (0-1) - quanto mais próximo de 1, melhor

## 🔧 Troubleshooting

### Drag & Drop não funciona
```bash
pip install tkinterdnd2
```

### Métodos Deep Learning falham
```bash
# Instale PyTorch
pip install torch torchvision

# Instale Kornia (para LoFTR)
pip install kornia
```

### GPU não detectada
- Verifique se PyTorch está instalado com suporte CUDA/ROCm
- Marque a opção "Tentar GPU" na interface
- Veja log para confirmação

### Imagens não aparecem no preview
- Verifique se PIL/Pillow está instalado
- Formatos suportados: JPG, PNG, BMP, TIF/TIFF

## 💡 Dicas

1. **Ordem importa**: Para panoramas, organize imagens da esquerda para direita
2. **Overlap**: Mantenha 30-50% de sobreposição entre imagens
3. **Iluminação**: Imagens com iluminação similar produzem melhores resultados
4. **Modo SCANS**: Use para documentos ou cenas planas
5. **Modo PANORAMA**: Use para paisagens e cenas 360°
6. **Comparação**: Execute "Todos" para identificar melhor método para seu caso
7. **Presets**: Crie presets para diferentes tipos de projetos (outdoor, indoor, docs, etc)

## 🏆 Melhores Práticas

### Para Paisagens
- Método: **LoFTR** ou **Seams Multiband**
- Modo: Panorama
- 30-40% overlap

### Para Documentos
- Método: **OpenCV Stitcher** ou **ECC Alignment**
- Modo: Scans
- 20-30% overlap

### Para Baixa Textura
- Método: **LoFTR** ou **ORB Homography**
- Modo: Panorama
- 40-50% overlap (mais overlap compensa baixa textura)

### Para Alta Velocidade
- Método: **SIFT Básico** ou **Blend Exposure**
- Modo: Qualquer
- Desabilitar GPU se não disponível

## 📝 Licença

Este software é fornecido "como está", sem garantias de qualquer tipo.

## 🤝 Contribuições

Melhorias são bem-vindas! Principais áreas:
- Novos métodos de stitching
- Otimizações de performance
- Melhorias na UI/UX
- Documentação

---

**Desenvolvido com** ❤️ **usando Python, OpenCV, Tkinter e PyTorch**
