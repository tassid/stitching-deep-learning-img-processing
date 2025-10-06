# 📊 Guia Completo do Sistema de Métricas

Sistema completo para avaliar e comparar algoritmos de image stitching com cálculo automático de métricas (PSNR, SSIM, MSE, entropia, nitidez).

---

## 🎯 O que foi implementado?

### ✅ 1. GUI Corrigida (`tk_stitcher_gui.py`)
- **Feedback visual claro** quando métricas são calculadas
- **Validação** de imagem de referência antes de calcular
- **Avisos informativos** quando checkbox está marcado mas referência não foi selecionada
- **Mensagens de erro** detalhadas se cálculo falhar

### ✅ 2. Módulo de Métricas (`metricas.py`)
Classe `MetricasImageStitching` com suporte a:
- **PSNR** (Peak Signal-to-Noise Ratio) em dB
- **SSIM** (Structural Similarity Index) de 0 a 1
- **MSE** (Mean Squared Error)
- **Entropia** (Shannon entropy) - medida de informação
- **Nitidez** (variância do Laplaciano) - detecta blur

**Funciona com ou sem scikit-image** (tem fallback)

### ✅ 3. Pipeline Automático (`pipeline.py`)
Sistema para executar **todos os algoritmos** em **todos os datasets** automaticamente:
- Descobre algoritmos disponíveis em `modules/`
- Processa múltiplos conjuntos de imagens
- Calcula métricas para cada resultado
- Salva tudo em CSV + JSON

### ✅ 4. Visualizador (`visualizar_resultados.py`)
Gera análises completas:
- **Tabelas LaTeX** (resumo e detalhada)
- **Gráficos** (PSNR, SSIM, tempo, scatter, heatmap)
- **Estatísticas** descritivas em TXT
- **Rankings** dos melhores resultados

---

## 📦 Instalação

### Dependências obrigatórias:
```bash
pip install opencv-python numpy pandas
```

### Dependências opcionais (recomendadas):
```bash
# Para métricas mais precisas
pip install scikit-image

# Para gerar gráficos
pip install matplotlib seaborn
```

---

## 🚀 Como Usar

### **Opção 1: GUI Interativa** (mais simples)

1. **Abrir a GUI:**
```bash
python tk_stitcher_gui.py
```

2. **Adicionar imagens de entrada:**
   - Clique em "Adicionar imagens..." ou "Adicionar pasta..."
   - Selecione pelo menos 2 imagens

3. **Configurar métricas:**
   - ✅ Marque checkbox **"Calcular métricas vs referência (PSNR/SSIM)"**
   - Clique em **"Escolher..."** ao lado de "Referência:"
   - Selecione a imagem de ground truth (ex: `result.jpg`)

4. **Executar:**
   - Clique em "▶ Executar"
   - Aguarde o processamento
   - Métricas aparecerão no log:
     ```
     📊 Calculando métricas vs referência...
     ✅ Métricas calculadas: PSNR=28.34 dB | SSIM=0.8234
     ```

5. **Verificar resultados:**
   - Imagem salva em: `results/pano_*.jpg`
   - Métricas salvas em: `results/pano_*.txt`
   - Log completo em: `results/metrics_log.txt`

---

### **Opção 2: Pipeline Automático** (comparar todos algoritmos)

#### 2.1. Preparar Dataset

**Baixar dataset do paper:**
```bash
git clone https://github.com/visionxiang/Image-Stitching-Dataset datasets/Image-Stitching-Dataset
```

**Estrutura esperada:**
```
datasets/
└── Image-Stitching-Dataset/
    └── Robust_Multiple_Registrations/
        ├── stop_sign_1/
        │   ├── input1.jpg
        │   ├── input2.jpg
        │   └── result.jpg  ← ground truth
        ├── graffiti_building_1/
        ├── cars_1/
        ├── ski_left/
        └── P2_front_of_window/
```

#### 2.2. Executar Pipeline

**Teste rápido** (apenas 1 conjunto):
```bash
python pipeline.py --test
```

**Executar todos** os conjuntos:
```bash
python pipeline.py
```

**Executar conjuntos específicos:**
```bash
python pipeline.py --conjuntos stop_sign_1 cars_1
```

**Especificar diretórios:**
```bash
python pipeline.py --datasets caminho/para/datasets --output caminho/para/resultados
```

#### 2.3. O que o pipeline faz?

1. ✅ Carrega todos os conjuntos de imagens
2. ✅ Executa **cada algoritmo disponível** em `modules/classic/` e `modules/deep/`
3. ✅ Calcula **todas as métricas** para cada resultado
4. ✅ Salva imagens em `resultados_pipeline/*.jpg`
5. ✅ Salva CSV com todas as métricas: `resultados_metricas_YYYYMMDD_HHMMSS.csv`
6. ✅ Salva backup JSON

**Exemplo de saída do pipeline:**
```
================================================================================
🚀 INICIANDO PIPELINE DE AVALIAÇÃO COMPLETA
================================================================================
📋 Conjuntos a processar: 5
🔧 Algoritmos a testar: 6
🧪 Total de experimentos: 30

[1/5] Conjunto: stop_sign_1
================================================================================
🔄 PROCESSANDO: stop_sign_1
================================================================================
📂 Carregando conjunto: datasets/.../stop_sign_1
   ✅ Ground truth encontrado

🔧 Executando: opencv_stitcher... ✅ (1.85s)
   📊 PSNR: 28.34 dB | SSIM: 0.8234
   📐 Tamanho: 1200x800 | Entropia: 7.234

🔧 Executando: orb_homography... ✅ (2.12s)
   📊 PSNR: 26.78 dB | SSIM: 0.7891
...
```

#### 2.4. Formato do CSV

```csv
conjunto,algoritmo,tempo_segundos,psnr_db,ssim,mse,entropia,nitidez,arquivo_saida,largura,altura
stop_sign_1,opencv_stitcher,1.85,28.34,0.8234,125.45,7.234,145.23,stop_sign_1_opencv_stitcher.jpg,1200,800
stop_sign_1,orb_homography,2.12,26.78,0.7891,156.32,7.156,138.90,stop_sign_1_orb_homography.jpg,1150,780
...
```

---

### **Opção 3: Visualizar Resultados**

Após executar o pipeline, gerar análises:

```bash
python visualizar_resultados.py resultados_pipeline/resultados_metricas_*.csv
```

**Arquivos gerados** em `resultados_pipeline/analises/`:

1. **📄 Tabelas LaTeX:**
   - `tabela_resumo.tex` - Médias por algoritmo
   - `tabela_detalhada.tex` - Todos os experimentos

2. **📊 Gráficos PNG (300 DPI):**
   - `grafico_psnr.png` - Comparação de PSNR por algoritmo
   - `grafico_ssim.png` - Comparação de SSIM por algoritmo
   - `grafico_tempo.png` - Tempo de execução (barras horizontais)
   - `grafico_scatter_trade_off.png` - Qualidade vs Velocidade
   - `heatmap_psnr.png` - PSNR por (conjunto, algoritmo)

3. **📝 Estatísticas:**
   - `estatisticas.txt` - Resumo textual com rankings

---

## 🧪 Teste Rápido

Para verificar se tudo está funcionando:

```bash
python teste_rapido_metricas.py
```

Este script:
- ✅ Testa imports de todos os módulos
- ✅ Cria imagens sintéticas
- ✅ Calcula métricas em diferentes cenários
- ✅ Valida resultados
- ✅ Mostra próximos passos

**Saída esperada:**
```
================================================================================
🧪 TESTE RÁPIDO DO SISTEMA DE MÉTRICAS
================================================================================

[1/5] Testando imports...
✅ metricas.py importado com sucesso
✅ pipeline.py importado com sucesso

[2/5] Criando imagens sintéticas de teste...
✅ Imagens sintéticas criadas em: test_temp_metricas

[3/5] Testando cálculo de métricas...
  📊 Teste A: Resultado perfeito (cópia exata)
     PSNR: inf dB (esperado: inf ou muito alto)
     SSIM: 1.0000 (esperado: ~1.0)
     MSE:  0.0000 (esperado: ~0.0)

  📊 Teste B: Resultado com ruído
     PSNR: 28.45 dB (esperado: 20-35 dB)
     SSIM: 0.9521 (esperado: 0.8-0.99)
     MSE:  92.34 (esperado: > 0)
...
✅ TESTE COMPLETO FINALIZADO!
```

---

## 📊 Interpretação das Métricas

### **PSNR (Peak Signal-to-Noise Ratio)**
- **Unidade:** dB (decibéis)
- **Range:** 0 a infinito (infinito = perfeito)
- **Interpretação:**
  - ✅ **> 30 dB** → Excelente qualidade
  - ✅ **25-30 dB** → Boa qualidade
  - ⚠️ **20-25 dB** → Qualidade aceitável
  - ❌ **< 20 dB** → Qualidade ruim

**Baseline do paper:**
- `stop_sign_1`: 19.49 dB (melhor resultado do paper)
- `graffiti_building_1`: 14.99 dB

**Meta:** Superar 19.49 dB!

### **SSIM (Structural Similarity Index)**
- **Unidade:** adimensional
- **Range:** 0 a 1 (1 = idêntico)
- **Interpretação:**
  - ✅ **0.90-1.00** → Excelente similaridade
  - ✅ **0.80-0.90** → Boa similaridade
  - ⚠️ **0.70-0.80** → Similaridade aceitável
  - ❌ **< 0.70** → Baixa similaridade

**Baseline do paper:**
- `stop_sign_1`: MS-SSIM = 0.6851

### **MSE (Mean Squared Error)**
- **Unidade:** pixel²
- **Range:** 0 a infinito (0 = perfeito)
- **Interpretação:**
  - ✅ **< 100** → Muito bom
  - ⚠️ **100-500** → Aceitável
  - ❌ **> 500** → Ruim

### **Entropia**
- **Unidade:** bits
- **Range:** 0 a ~8 (para imagens 8-bit)
- **Interpretação:**
  - Maior = mais informação preservada
  - Típico: 6.5 - 7.5 para imagens naturais

### **Nitidez**
- **Unidade:** variância
- **Range:** 0 a infinito
- **Interpretação:**
  - ✅ **> 100** → Imagem nítida
  - ⚠️ **50-100** → Levemente borrada
  - ❌ **< 50** → Muito borrada

---

## 🔧 Troubleshooting

### **Problema 1: PSNR e SSIM sempre `nan`**

**Causa:** Checkbox não marcado OU referência não selecionada

**Solução:**
1. ✅ Marque checkbox "Calcular métricas vs referência"
2. ✅ Clique em "Escolher..." e selecione imagem de ground truth
3. Execute novamente

---

### **Problema 2: "scikit-image não encontrado"**

**Causa:** Biblioteca opcional não instalada

**Solução:**
```bash
pip install scikit-image
```

**Alternativa:** Sistema funciona com fallback (métricas menos precisas)

---

### **Problema 3: "Dataset não encontrado"**

**Causa:** Dataset não está no caminho esperado

**Solução:**
```bash
# Baixar dataset
git clone https://github.com/visionxiang/Image-Stitching-Dataset datasets/Image-Stitching-Dataset

# Ou especificar caminho customizado
python pipeline.py --datasets caminho/para/seu/dataset
```

---

### **Problema 4: Gráficos não são gerados**

**Causa:** matplotlib não instalado

**Solução:**
```bash
pip install matplotlib seaborn
```

---

### **Problema 5: Erro "Falha ao calcular métricas"**

**Possíveis causas:**
- Imagem de referência corrompida
- Dimensões muito diferentes (sistema redimensiona automaticamente)
- Formato de imagem não suportado

**Solução:**
- Verifique se referência é uma imagem válida
- Tente com outro ground truth
- Confira logs detalhados na GUI

---

## 📈 Exemplo de Workflow Completo

```bash
# 1. Testar sistema
python teste_rapido_metricas.py

# 2. Baixar dataset
git clone https://github.com/visionxiang/Image-Stitching-Dataset datasets/Image-Stitching-Dataset

# 3. Executar pipeline de teste (1 conjunto)
python pipeline.py --test

# 4. Se OK, executar pipeline completo
python pipeline.py

# 5. Visualizar resultados
python visualizar_resultados.py resultados_pipeline/resultados_metricas_*.csv

# 6. Abrir análises
cd resultados_pipeline/analises
# Ver estatisticas.txt, gráficos PNG, tabelas LaTeX
```

---

## 📊 Estrutura de Arquivos Gerados

```
projeto/
├── metricas.py                    ← Módulo de cálculo
├── pipeline.py                    ← Pipeline automático
├── visualizar_resultados.py       ← Gerador de análises
├── teste_rapido_metricas.py       ← Script de teste
├── tk_stitcher_gui.py            ← GUI (corrigida)
├── GUIA_METRICAS.md              ← Este guia
│
├── datasets/
│   └── Image-Stitching-Dataset/
│       └── Robust_Multiple_Registrations/
│           ├── stop_sign_1/
│           ├── graffiti_building_1/
│           ├── cars_1/
│           ├── ski_left/
│           └── P2_front_of_window/
│
├── results/                       ← Resultados da GUI
│   ├── pano_*.jpg
│   ├── pano_*.txt
│   └── metrics_log.txt
│
└── resultados_pipeline/           ← Resultados do pipeline
    ├── resultados_metricas_*.csv  ← CSV principal
    ├── resultados_metricas_*.json
    ├── stop_sign_1_*.jpg          ← Imagens resultado
    ├── graffiti_building_1_*.jpg
    └── analises/                  ← Análises geradas
        ├── tabela_resumo.tex
        ├── tabela_detalhada.tex
        ├── grafico_psnr.png
        ├── grafico_ssim.png
        ├── grafico_tempo.png
        ├── grafico_scatter_trade_off.png
        ├── heatmap_psnr.png
        └── estatisticas.txt
```

---

## 🎓 Referências

**Dataset:**
- Robust Image Stitching with Multiple Registrations
- Zhang et al. (2018)
- https://github.com/visionxiang/Image-Stitching-Dataset

**Métricas:**
- PSNR: [Wikipedia](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)
- SSIM: Wang et al. (2004) - "Image Quality Assessment"
- Entropy: Shannon (1948) - Information Theory

---

## ✅ Checklist de Verificação

Antes de executar o pipeline completo:

- [ ] ✅ `teste_rapido_metricas.py` executou sem erros
- [ ] ✅ Dataset baixado e estrutura verificada
- [ ] ✅ Dependências instaladas (`opencv-python`, `numpy`, `pandas`)
- [ ] ✅ (Opcional) scikit-image instalado
- [ ] ✅ (Opcional) matplotlib instalado
- [ ] ✅ Algoritmos funcionando em `modules/`

---

## 🆘 Suporte

**Logs úteis:**
- GUI: Veja seção "Log" na interface
- Pipeline: Saída do terminal tem progresso detalhado
- Métricas: `results/metrics_log.txt`

**Comandos de diagnóstico:**
```bash
# Verificar imports
python -c "from metricas import MetricasImageStitching; print('OK')"
python -c "from pipeline import PipelineImageStitching; print('OK')"

# Verificar dependências
python -c "import cv2, numpy, pandas; print('Deps OK')"
python -c "from skimage.metrics import structural_similarity; print('scikit-image OK')"
python -c "import matplotlib; print('matplotlib OK')"

# Listar algoritmos descobertos
python -c "from pipeline import PipelineImageStitching; p = PipelineImageStitching(); print(list(p.algoritmos.keys()))"
```

---

## 🚀 Próximos Passos Recomendados

1. ✅ Executar `teste_rapido_metricas.py`
2. ✅ Baixar dataset completo
3. ✅ Testar GUI com 1 conjunto
4. ✅ Executar pipeline com `--test`
5. ✅ Executar pipeline completo
6. ✅ Gerar visualizações
7. ✅ Analisar resultados e identificar melhor algoritmo
8. ✅ Usar melhores resultados no TCC!

---

**✨ Sistema pronto para uso! Boa sorte com seu TCC! 🎓**
