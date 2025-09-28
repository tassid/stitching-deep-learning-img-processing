# stitching-deep-learning-img-processing — Clássicos

Scripts para **stitching**/mosaico e **blending** com técnicas clássicas (OpenCV + PDI), usando imagens locais **ou** frames de datasets da **Hugging Face**. Foco em sequências de pomares e vídeos, com ferramentas para avaliar qualidade (SSIM/MSE/PSNR).

---

## Sumário
- [Requisitos](#requisitos)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Como executar (padrão)](#como-executar-padrão)
- [Catálogo de módulos](#catálogo-de-módulos)
  - [1) Stitcher OpenCV (baseline)](#1-stitcher-opencv-baseline)
  - [2) SIFT + RANSAC + feather](#2-sift--ransac--feather)
  - [3) Pipeline de Features + Homografia](#3-pipeline-de-features--homografia)
  - [4) Exposure Fusion (Mertens)](#4-exposure-fusion-mertens)
  - [5) Fusão Wavelet](#5-fusão-wavelet)
  - [6) Seams + ExposureComp + Multiband](#6-seams--exposurecomp--multiband)
  - [7) Detectores Harris/FAST (demo)](#7-detectores-harrisfast-demo)
  - [8) Blend Multibanda (Laplaciana “pura”)](#8-blend-multibanda-laplaciana-pura)
  - [9) Exposure Compensation (isolada)](#9-exposure-compensation-isolada)
  - [10) Poisson / Gradient-Domain](#10-poisson--gradient-domain)
  - [11) Fusão Multi-Focal](#11-fusão-multi-focal)
  - [12) Fusão Híbrida (Mertens + foco)](#12-fusão-híbrida-mertens--foco)
  - [13) Alinhamento por ECC](#13-alinhamento-por-ecc)
  - [Métricas (SSIM/MSE/PSNR)](#métricas-ssimmsepsnr)
- [Boas práticas de sobreposição](#boas-práticas-de-sobreposição)
- [Erros comuns](#erros-comuns)
- [Referências](#referências)

---

## Requisitos

- **Python 3.11+** (recomendado usar *venv*)
- Pacotes principais:
  - `opencv-contrib-python==4.10.*`
  - `datasets`, `pillow`, `huggingface_hub[hf_xet]`
- Pacotes extras por módulo:
  - `PyWavelets` (fusão wavelet)
  - `scikit-image` (SSIM/PSNR em `metrics.py`)

Instalação rápida (Windows/PowerShell):
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip wheel setuptools
pip install "opencv-contrib-python==4.10.*" datasets pillow "huggingface_hub[hf_xet]" PyWavelets scikit-image
```

Validação:
```powershell
python -c "import cv2; print(cv2.__version__); print('detail:', hasattr(cv2,'detail')); print('PyRotationWarper:', hasattr(cv2,'PyRotationWarper'))"
```

---

## Estrutura do projeto

```
modules/
  classic/
    1_stitcher_basico.py
    2_manual_sift_ransac_feather.py
    3_features_homography.py
    4_blend_exposure.py
    5_blend_wavelet.py
    6_seams_exposure_multiband.py
    7_detectors_harris_fast.py
    8_blend_laplacian.py
    9_exposure_comp_only.py
    10_blend_poisson.py
    11_blend_multifocal.py
    12_blend_hybrid.py
    13_ecc_alignment.py
    metrics.py
```

---

## Como executar (padrão)

### Arquivos locais
```powershell
python -m modules.classic.<script> --imgs a.jpg b.jpg c.jpg --out pano.jpg
```

### Dataset Hugging Face
```powershell
# se o dataset for privado: huggingface-cli login
python -m modules.classic.<script> --hf miroslavjaros/Video-To-Dataset-Orchard-Apple `
  --split train --count 24 --offset 300 --out pano.jpg
```

---

## Catálogo de módulos

### 1) Stitcher OpenCV (baseline)
**Arquivo:** `modules/classic/1_stitcher_basico.py`  
**Descrição:** mosaico automático via **OpenCV Stitcher** (modo `panorama` ou `scans`). Útil como baseline rápido e para checar sobreposição/warpers.  
**Exemplos:**
```powershell
python -m modules.classic.1_stitcher_basico --imgs a.jpg b.jpg c.jpg --modo panorama --out pano.jpg
python -m modules.classic.1_stitcher_basico --hf <dataset> --split train --count 24 --offset 300 --modo scans --out pano.jpg
```

---

### 2) SIFT + RANSAC + feather
**Arquivo:** `modules/classic/2_manual_sift_ransac_feather.py`  
**Descrição:** SIFT + KNN (ratio test), homografia com **USAC_MAGSAC** e **feather** no overlap.  
**Exemplo:**
```powershell
python -m modules.classic.2_manual_sift_ransac_feather --imgs a.jpg b.jpg c.jpg --ratio 0.75 --out pano_manual.jpg
```

---

### 3) Pipeline de Features + Homografia
**Arquivo:** `modules/classic/3_features_homography.py`  
**Descrição:** pipeline genérico com **SIFT/ORB/AKAZE** + **BF/FLANN** + **RANSAC**; ponto de partida para ablações.  
**Exemplo:**
```powershell
python -m modules.classic.3_features_homography --imgs a.jpg b.jpg c.jpg --detector sift --matcher flann --out pano_feat.jpg
```

---

### 4) Exposure Fusion (Mertens)
**Arquivo:** `modules/classic/4_blend_exposure.py`  
**Descrição:** **contraste + saturação + boa exposição** (Mertens et al.) com blend **laplaciano**.  
**Exemplo:**
```powershell
python -m modules.classic.4_blend_exposure --imgs a.jpg b.jpg c.jpg --w_contrast 1 --w_saturation 1 --w_exposure 1 --out pano_mertens.jpg
```

---

### 5) Fusão Wavelet
**Arquivo:** `modules/classic/5_blend_wavelet.py`  
**Descrição:** decomposição multi-escala (wavelets) e regra por banda (máximo/energia).  
**Exemplo:**
```powershell
python -m modules.classic.5_blend_wavelet --imgs a.jpg b.jpg c.jpg --wavelet db2 --levels 3 --rule max --out pano_wavelet.jpg
```

---

### 6) Seams + ExposureComp + Multiband
**Arquivo:** `modules/classic/6_seams_exposure_multiband.py`  
**Descrição:** **seam finding** (DP/GraphCut c/ fallback), **compensação de exposição** (gain/gain-blocks) e **multiband blending** (`cv2.detail`).  
**Exemplo:**
```powershell
python -m modules.classic.6_seams_exposure_multiband --imgs a.jpg b.jpg c.jpg --seam dp --bands 5 --out pano_seams.jpg
```

---

### 7) Detectores Harris/FAST (demo)
**Arquivo:** `modules/classic/7_detectors_harris_fast.py`  
**Descrição:** visualização e comparação de detectores **Harris** e **FAST** (parâmetros de *threshold*, *nonmax*, etc.).  
**Exemplo:**
```powershell
python -m modules.classic.7_detectors_harris_fast --imgs a.jpg --detector harris --alpha 0.04 --max_kp 2000
```

---

### 8) Blend Multibanda (Laplaciana “pura”)
**Arquivo:** `modules/classic/8_blend_laplacian.py`  
**Descrição:** **pirâmides Laplacianas** (Burt–Adelson); mistura por frequência apenas na região de sobreposição.  
**Exemplo:**
```powershell
python -m modules.classic.8_blend_laplacian --imgs a.jpg b.jpg c.jpg --levels 5 --out pano_lap.jpg
```

---

### 9) Exposure Compensation (isolada)
**Arquivo:** `modules/classic/9_exposure_comp_only.py`  
**Descrição:** equalização fotométrica (ganho/viñeta) antes do blend; *feather* simples para evitar cortes duros.  
**Exemplo:**
```powershell
python -m modules.classic.9_exposure_comp_only --imgs a.jpg b.jpg c.jpg --comp gain_blocks --out pano_exposure.jpg
```

---

### 10) Poisson / Gradient-Domain
**Arquivo:** `modules/classic/10_blend_poisson.py`  
**Descrição:** **seamless cloning** (edição por Poisson) na área de overlap; preserva gradientes e ajuda em costuras difíceis.  
**Exemplo:**
```powershell
python -m modules.classic.10_blend_poisson --imgs a.jpg b.jpg c.jpg --mode mixed --out pano_poisson.jpg
```

---

### 11) Fusão Multi-Focal
**Arquivo:** `modules/classic/11_blend_multifocal.py`  
**Descrição:** escolhe, por pixel, a fonte **mais nítida** (variância do Laplaciano/Tenengrad) e mistura por pirâmide.  
**Exemplo:**
```powershell
python -m modules.classic.11_blend_multifocal --imgs a.jpg b.jpg c.jpg --focus lapvar --levels 5 --out pano_multifocal.jpg
```

---

### 12) Fusão Híbrida (Mertens + foco)
**Arquivo:** `modules/classic/12_blend_hybrid.py`  
**Descrição:** pesos híbridos (contraste/saturação/exposição **+** foco), suavizados multi-escala; blend **laplaciano**.  
**Exemplo:**
```powershell
python -m modules.classic.12_blend_hybrid --imgs a.jpg b.jpg c.jpg --w_contrast 1 --w_saturation 1 --w_exposure 1 --w_focus 0.5 --out pano_hybrid.jpg
```

---

### 13) Alinhamento por ECC
**Arquivo:** `modules/classic/13_ecc_alignment.py`  
**Descrição:** registro incremental por **ECC** (translation/euclidean/affine/homography), bom para cenas **planares**; *feather* no overlap.  
**Exemplo:**
```powershell
python -m modules.classic.13_ecc_alignment --imgs a.jpg b.jpg c.jpg --warp affine --out pano_ecc.jpg
```

---

### Métricas (SSIM/MSE/PSNR)
**Arquivo:** `modules/classic/metrics.py`  
**Descrição:** compara panoramas (ref + candidatos ou diretório) e gera **CSV** + **README.md** com tabela.  
**Exemplos:**
```powershell
# referência vs candidatos
python -m modules.classic.metrics --ref pano_ref.jpg --imgs pano_A.jpg pano_B.jpg --mask-nonzero --psnr --outdir reports_metrics

# todas as combinações da pasta
python -m modules.classic.metrics --dir .\outs --align resize --mask-nonzero --outdir reports_metrics
```

---

## Boas práticas de sobreposição

- **Frames consecutivos** em datasets de vídeo → mais *inliers* e menos falhas.  
- **Overlap** ~30–50% entre vizinhos e manter **ordem L→R** (ou temporal).  
- **Redimensionamento**: `--min_side 900–1200` balanceia robustez/tempo.  
- Para variação de iluminação: **exposure compensation** + **multiband** ou **Poisson**.  
- Para cenas planares (documentos/fachadas): **ECC** (`--warp affine/homography`) e warper **plane**.

---

## Erros comuns

- **Stitcher status 1** (`NEED_MORE_IMGS`): pouca sobreposição → aumente `--count`/ajuste `--offset`.  
- **Stitcher status 3**: ajuste de câmera falhou → use `--modo scans`, troque warper (`cylindrical`/`plane`), mude SIFT↔ORB/AKAZE.  
- **`ImportError: Pillow`** ao carregar HF → `pip install pillow`.  
- **Aviso de symlink no HF cache (Windows)** → ative **Developer Mode** ou oculte com `setx HF_HUB_DISABLE_SYMLINKS_WARNING 1`.

---

## Referências

- **Panoramas por features**: Brown, M.; Lowe, D. (2007). *Automatic Panoramic Image Stitching using Invariant Features*.
- **Pirâmides Laplacianas**: Burt, P.; Adelson, E. (1983). *A Multiresolution Spline With Application to Image Mosaics*.
- **Exposure Fusion**: Mertens, T.; Kautz, J.; Van Reeth, F. (2007). *Exposure Fusion*.
- **Poisson/Seamless cloning**: Pérez, P.; Gangnet, M.; Blake, A. (2003). *Poisson Image Editing*.
- **Harris**: Harris, C.; Stephens, M. (1988). *A Combined Corner and Edge Detector*.
- **FAST**: Rosten, E.; Drummond, T. (2006/2010). *Machine learning for high-speed corner detection* / *Faster and better: a machine learning approach to corner detection*.
- **Wavelets (base)**: Mallat, S. (1999). *A Wavelet Tour of Signal Processing*.
- **Surveys de registro/blending**: Zitová, B.; Flusser, J. (2003). *Image registration methods: a survey*. Sunkavalli, K. et al. (2010); He, K.; Sun, J. (2012).

---

> **Dica final:** para datasets HF com frames de vídeo, varra `--offset` em passos e mantenha `--count` ≥ 16–32 para garantir sobreposição. Combine com `metrics.py` para comparar resultados entre técnicas.
