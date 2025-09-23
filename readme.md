# 1\_stitcher\_basico — Documentação

> **Projeto:** stitching-deep-learning-img-processing
> **Script principal:** `1_stitcher_basico.py`
> **Objetivo:** Montar panoramas/varreduras a partir de imagens locais **ou** frames de datasets da Hugging Face.

---

## Sumário

* [Visão geral](#visão-geral)
* [Requisitos](#requisitos)
* [Instalação rápida (Windows/PowerShell)](#instalação-rápida-windowspowershell)
* [Uso](#uso)

  * [Imagens locais](#imagens-locais)
  * [Dataset Hugging Face](#dataset-hugging-face)
  * [Varrendo offsets (PowerShell)](#varrendo-offsets-powershell)
* [Parâmetros do script](#parâmetros-do-script)
* [Modos e Warpers](#modos-e-warpers)
* [Exemplos prontos](#exemplos-prontos)
* [Diagnóstico e erros comuns](#diagnóstico-e-erros-comuns)
* [Boas práticas de sobreposição](#boas-práticas-de-sobreposição)
* [Dicas de performance](#dicas-de-performance)
* [Troubleshooting do Windows](#troubleshooting-do-windows)

---

## Visão geral

O `1_stitcher_basico.py` cria panoramas com o **OpenCV Stitcher** a partir de uma lista de imagens. Ele suporta: (1) **arquivos locais** e (2) **datasets Hugging Face** via `datasets.load_dataset`, convertendo imagens PIL→OpenCV (RGB→BGR). O script tenta configurar um warper (spherical) quando disponível; se sua versão do OpenCV não suportar `setWarper`, o código segue com o padrão, evitando quebras.

---

## Requisitos

* **Python 3.11+** (3.12 também funciona; 3.13 pode ter dependências ainda imaturas)
* **Virtualenv recomendado**
* Pacotes:

  * `opencv-contrib-python` (habilita `PyRotationWarper`/`setWarper`)
  * `datasets`, `pillow`, `huggingface_hub[hf_xet]`

```powershell
# (opcional) criar venv
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# limpar e instalar opencv contrib
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install -U pip wheel setuptools
pip install "opencv-contrib-python==4.10.*"

# deps HF/datasets
pip install datasets pillow "huggingface_hub[hf_xet]"

# validar
python -c "import cv2; print(cv2.__version__); print('has PyRotationWarper:', hasattr(cv2,'PyRotationWarper'))"
```

---

## Uso

### Imagens locais

```powershell
python .\1_stitcher_basico.py --imgs a.jpg b.jpg c.jpg --modo panorama --out pano.jpg
```

### Dataset Hugging Face

1. (Se privado) `huggingface-cli login`
2. Exemplo com **LivingOptics/hyperspectral-orchard** (usa RGB do dataset):

```powershell
python .\1_stitcher_basico.py --hf LivingOptics/hyperspectral-orchard --split train --count 5 --out pano_orchard.jpg
```

> Obs.: Esse dataset não necessariamente tem frames consecutivos; panoramização pode falhar por falta de sobreposição.

3. Exemplo com **miroslavjaros/Video-To-Dataset-Orchard-Apple** (frames de vídeo):

```powershell
# escolha blocos consecutivos; ajuste offset e count
python .\1_stitcher_basico.py --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train --count 24 --offset 300 --modo scans --out pano_300.jpg
```

### Varrendo offsets (PowerShell)

Gere vários panoramas para encontrar trechos com boa sobreposição:

```powershell
for ($o=0; $o -le 1200; $o+=50) {
  python .\1_stitcher_basico.py --hf miroslavjaros/Video-To-Dataset-Orchard-Apple `
    --split train --count 24 --offset $o --modo scans --out "pano_$o.jpg"
}
```

---

## Parâmetros do script

```text
--imgs   Lista de arquivos locais (alternativa a --hf)
--hf     Nome do dataset na Hugging Face (ex.: LivingOptics/hyperspectral-orchard)
--split  Split do dataset (padrão: train)
--count  Quantidade de imagens (padrão: 5)
--offset Índice inicial dentro do split (padrão: 0)
--shuffle Embaralhar (evite para sequências)
--modo   panorama | scans (padrão: panorama)
--try_gpu (placeholder, para builds com CUDA)
--out    Nome do arquivo de saída (padrão: pano_stitcher.jpg)
```

---

## Modos e Warpers

* **Modo**

  * `panorama`: cenas com parallax e rotação da câmera; use warpers não planares.
  * `scans`: cenas quase planares (documentos, fachadas, fileiras em linha reta).
* **Warper** (auto-fallback no script)

  * `spherical`: bom para panoramas largos; requer `opencv-contrib-python`.
  * `cylindrical`: alternativa robusta se spherical falhar.
  * `plane`: para `--modo scans` ou quando a cena for essencialmente plana.

Para ajustar manualmente, edite no `make_stitcher` a linha do `Warper("spherical", 1.0)` para `"cylindrical"` ou `"plane"`.

---

## Exemplos prontos

**1) Panorama simples com arquivos locais**

```powershell
python .\1_stitcher_basico.py --imgs .\imgs\p1.jpg .\imgs\p2.jpg .\imgs\p3.jpg --modo panorama --out pano_local.jpg
```

**2) Teste sintético (garante overlap) a partir de uma imagem**

```powershell
python - <<'EOF'
import cv2
im=cv2.imread('uma_foto.jpg'); h,w=im.shape[:2]
crops=[im[:,0:int(0.7*w)], im[:,int(0.15*w):int(0.85*w)], im[:,int(0.3*w):w]]
for i,c in enumerate(crops): cv2.imwrite(f'crop{i}.jpg',c)
EOF
python .\1_stitcher_basico.py --imgs crop0.jpg crop1.jpg crop2.jpg --modo panorama --out pano_sintetico.jpg
```

**3) Frames de vídeo (sequência mais longa)**

```powershell
python .\1_stitcher_basico.py --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train --count 32 --offset 450 --modo scans --out pano_450.jpg
```

---

## Diagnóstico e erros comuns

* **status 1 — ERR\_NEED\_MORE\_IMGS**: falta de sobreposição/feature. Use frames **consecutivos** e aumente `--count`.
* **status 3 — ERR\_CAMERA\_PARAMS\_ADJUST\_FAIL**: bundle adjustment falhou. Tente `--modo scans`, troque warper para `cylindrical`/`plane`, reduza `setPanoConfidenceThresh` (no código) ou reordene as imagens.
* **`AttributeError: setWarper`**: instale `opencv-contrib-python`. O script já faz fallback se ausente.
* **`ImportError: install 'Pillow'`**: `pip install pillow` (o `datasets` usa Pillow para codificar imagens).

Log útil:

```powershell
python -c "import cv2; print(cv2.getBuildInformation())" | Select-String -Pattern Stitching
```

---

## Boas práticas de sobreposição

* **Ordem**: mantenha a sequência L→R (ou R→L) ou na ordem temporal dos frames.
* **Overlap**: \~30–50% entre imagens adjacentes.
* **Textura**: evite áreas totalmente uniformes (céu, chão homogêneo).
* **Resolução**: mínimo \~800–1000 px no menor lado para features suficientes.

---

## Dicas de performance

* Ative **Developer Mode** no Windows ou ignore aviso de symlink do HF cache:

```powershell
# ativar aviso off (persistente)
setx HF_HUB_DISABLE_SYMLINKS_WARNING 1
```

* Instale `huggingface_hub[hf_xet]` para downloads mais rápidos quando o repo usa Xet.
* Se memória for um problema, reduza `--count` e processe em blocos.

---

## Troubleshooting do Windows

* **PowerShell vs CMD**: Paths com espaço precisam de aspas. No PowerShell, prefixe `./` para scripts locais.
* **Permissões**: Se usar admin para symlinks, reabra o terminal como Administrador.
* **Ambiente**: mantenha o venv ativo (`.\.venv\Scripts\Activate.ps1`).

---

## Script de referência (versão atual)

O arquivo `1_stitcher_basico.py` já está preparado com:

* Fallback automático quando `setWarper` não existe;
* Carregamento de imagens locais **ou** via Hugging Face (`datasets`);
* Conversão segura PIL→OpenCV.

> Qualquer ajuste de warper/limiar pode ser feito na função `make_stitcher`. Se quiser, posso incluir flags para escolher `--warper spherical|cylindrical|plane` e `--conf 0.1..0.9` pelo CLI.

---

# Estrutura do Projeto (multi-modelos)

```
stitching-deep-learning-img-processing/
├─ classic/
│  ├─ stitcher_opencv/ (script atual + warpers)
│  ├─ features_homography/ (SIFT/ORB/AKAZE + RANSAC + warping)
│  ├─ ecc_alignment/ (ECC motion, planar/scans)
│  └─ utils/
├─ deep/
│  ├─ loftr/ (correspondência densa sem detectores)
│  ├─ superpoint_superglue/ (pontos + matching via GNN)
│  ├─ deep_homography/ (CNN p/ H4pt/8pt / differentiable RANSAC)
│  └─ opticalflow_warp/ (RAFT/PWCNet p/ warping incremental)
├─ datasets/
│  ├─ registry.yaml (mapeia HF/local → loaders)
│  ├─ loaders.py (interfaces padronizadas)
│  └─ samples/ (amostras mínimas p/ smoke tests)
├─ benchmarks/
│  ├─ run_bench.py (loop de offsets/counts)
│  ├─ metrics.py (inliers, reproj err, SSIM em overlaps, seam score)
│  └─ reports/ (CSV/MD com resultados)
├─ docs/ (este README + FAQ + guias)
└─ pyproject.toml / requirements.txt
```

## Roadmap de Técnicas

* **Clássicas**: OpenCV Stitcher (panorama/scans), *feature-based* (SIFT/ORB/AKAZE + FLANN/BRUTE + RANSAC), ECC para casos planares/escaneados.
* **Deep Learning**: **LoFTR** (global dense matches), **SuperPoint+SuperGlue** (detecção+matching robusto), **Deep Homography** (estimativa direta H), **Optical Flow** (RAFT/PWCNet) para warping incremental.
* **Pós-processamento**: equalização de exposição, compensação de ganho, *multi-band blending*, corte e retificação.

## Convenções de CLI (todas as técnicas)

* `--source` (`imgs`, `hf`), `--split/--count/--offset` (datasets), `--order` (L→R / R→L / por timestamp), `--mode` (`panorama|scans`), `--warper` (`auto|spherical|cylindrical|plane`), `--conf` (0–1), `--out`.
* **Deep**: `--model` (loftr|sp\_sg|deepH|raft), `--weights`, `--resize` (lado menor), `--device` (cpu|cuda), `--fp16`.

## Datasets (registry.yaml exemplo)

```yaml
miroslav_apple:
  type: hf
  name: miroslavjaros/Video-To-Dataset-Orchard-Apple
  split: train
  loader: video_frames
living_optics_orchard:
  type: hf
  name: LivingOptics/hyperspectral-orchard
  split: train
  loader: rgb_only
local_imgs:
  type: local
  glob: imgs/*.jpg
```

## Métricas e Avaliação

* **Matching**: #correspondências, taxa de *inliers* (RANSAC), erro médio de reprojeção (px).
* **Qualidade do mosaico**: SSIM/PSNR nas áreas de sobreposição, *seam cut energy* (gradiente), variação de exposição.
* **Taxa de sucesso**: % runs com status OK vs (1) NEED\_MORE\_IMGS e (3) CAMERA\_PARAMS\_FAIL.
* **Tempo**: por estágio (detecção, matching, ajuste, warping, blending).

## Benchmark Automatizado

* Script `benchmarks/run_bench.py` varre `--offset` em passos (ex.: 50) e `--count` (8, 16, 24, 32), salva **CSV** com métricas e gera um **MD** com tabela-resumo e thumbs dos panoramas.
* Exemplo (PowerShell):

```powershell
for ($o=0; $o -le 1500; $o+=50) {
  python benchmarks/run_bench.py --tech sp_sg \
    --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train \
    --count 24 --offset $o --mode scans --out reports/sp_sg_$o.jpg
}
```

## Guia Rápido por Técnica

* **Stitcher OpenCV**: rápido, baseline; bom para validar pipeline.
* **Features+H**: mais controle (SIFT/ORB/AKAZE); ajuste fino de *ratio test*, RANSAC, warper.
* **ECC**: cenas planares, documentos, fachadas → `--mode scans --warper plane`.
* **LoFTR**: robusto em textura/frutas repetitivas; útil quando detectores falham.
* **SuperPoint+SuperGlue**: excelente precisão nos *matches*; requer GPU para velocidade.
* **Deep Homography**: útil para grandes overlaps com pouca textura; checar generalização.
* **Optical Flow (RAFT/PWCNet)**: bom para varrer sequências longas; cuidado com *drift*.

## FAQ (inicial)

**Q: Status 1/3 o tempo todo em pomares?**
A: Use frames **consecutivos**, aumente `--count`, teste `--mode scans` e warpers `cylindrical/plane`.

**Q: Por que LoFTR/SuperGlue?**
A: Melhoram *matching* em padrões repetitivos (folhas/frutos), aumentando *inliers* e estabilidade do bundle.

**Q: Como comparar técnicas?**
A: Rode o `run_bench.py` em mesma janela de offsets/counts e compare a tabela de métricas (+ SSIM em overlaps).

**Q: Preciso de GPU?**
A: Clássicos rodam bem em CPU; LoFTR/SuperGlue/RAFT são muito mais rápidos em CUDA.

## Próximos Passos

* Adicionar `--warper` e `--conf` ao CLI do script clássico.
* Implementar `datasets/loaders.py` com interface comum (local/HF).
* Porta das técnicas deep (checkpoints + scripts).
* `benchmarks/run_bench.py` + `metrics.py` + geração de relatório MD/CSV.

---

# Entry Points e Esqueletos (o Stitcher básico é **um** módulo)

> O **projeto** é `stitching-deep-learning-img-processing` (tema do TCC). O `1_stitcher_basico.py` é **apenas um dos scripts** dentro de `classic/`. Abaixo, adiciono um **dispatcher principal**, além de esqueletos para `datasets/` e `benchmarks/`.

## `main.py` — Dispatcher do projeto

```python
# main.py — ponto de entrada do TCC
# uso: python main.py classic.stitcher --imgs a.jpg b.jpg c.jpg --modo panorama
#      python main.py classic.stitcher --hf miroslavjaros/Video-To-Dataset-Orchard-Apple --split train --count 24 --offset 300 --modo scans

import importlib, sys

def main():
    if len(sys.argv) < 2:
        print("use: python main.py <module> [args]
ex.: python main.py classic.stitcher --help")
        sys.exit(1)
    mod_name = sys.argv[1]  # p.ex. classic.stitcher
    try:
        mod = importlib.import_module(f"modules.{mod_name}")
    except Exception as e:
        print(f"não encontrei modules/{mod_name.replace('.', '/')}.py — erro: {e}")
        sys.exit(1)
    # repassar os argumentos remanescentes para o módulo
    sys.argv = [mod_name] + sys.argv[2:]
    if hasattr(mod, "cli_main"):
        sys.exit(mod.cli_main())
    print("módulo não expõe cli_main()")
    sys.exit(1)

if __name__ == "__main__":
    main()
```

## Estrutura sugerida de pacotes

```
modules/
├─ classic/
│  ├─ __init__.py
│  ├─ stitcher.py           # (nova versão do stitcher com flags --warper/--conf)
│  └─ features_homography.py
├─ deep/
│  ├─ __init__.py
│  ├─ loftr.py
│  └─ superpoint_superglue.py
├─ datasets/
│  ├─ __init__.py
│  └─ loaders.py
└─ benchmarks/
   ├─ __init__.py
   └─ run_bench.py
```

---

## `modules/classic/stitcher.py` — (substitui o antigo script básico)

```python
# modules/classic/stitcher.py
# uso: python main.py classic.stitcher --imgs ... OU --hf <dataset>
import cv2, argparse, sys
from typing import List

def pil_to_bgr(pil_img):
    import numpy as np
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def load_from_hf(name: str, split: str, count: int, offset: int, shuffle: bool) -> List:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("Instale 'datasets' e 'Pillow': pip install datasets pillow")
    ds = load_dataset(name, split=split)
    if shuffle:
        ds = ds.shuffle(seed=42)
    end = min(len(ds), offset + count)
    if end - offset < 2:
        sys.exit(f"Preciso de >=2 imagens; split '{split}' tem {len(ds)} registros.")
    imgs = []
    for rec in ds.select(range(offset, end)):
        img = rec.get("image") or rec.get("rgb") or rec[list(rec.keys())[0]]
        try:
            imgs.append(pil_to_bgr(img))
        except Exception:
            if isinstance(img, str):
                im = cv2.imread(img)
                if im is not None:
                    imgs.append(im)
    if len(imgs) < 2:
        sys.exit("Não consegui ler imagens RGB desse dataset/split.")
    return imgs

def make_stitcher(modo: str, warper: str|None, conf: float):
    mode = cv2.Stitcher_PANORAMA if modo == "panorama" else cv2.Stitcher_SCANS
    st = cv2.Stitcher_create(mode)
    try:
        st.setPanoConfidenceThresh(conf)
    except Exception:
        pass
    if warper and warper != "auto":
        try:
            Warper = getattr(cv2, "PyRotationWarper", None)
            if Warper and hasattr(st, "setWarper"):
                st.setWarper(Warper(warper, 1.0))
        except Exception:
            pass
    return st

def cli_main():
    p = argparse.ArgumentParser(prog="classic.stitcher")
    gsrc = p.add_mutually_exclusive_group(required=True)
    gsrc.add_argument("--imgs", nargs="+", help="arquivos locais")
    gsrc.add_argument("--hf", help="dataset HF, ex: miroslavjaros/Video-To-Dataset-Orchard-Apple")
    p.add_argument("--split", default="train")
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--modo", choices=["panorama","scans"], default="panorama")
    p.add_argument("--warper", choices=["auto","spherical","cylindrical","plane"], default="auto")
    p.add_argument("--conf", type=float, default=0.5, help="limiar de confiança do pano (0–1)")
    p.add_argument("--out", default="pano_stitcher.jpg")
    a = p.parse_args()

    imgs = [cv2.imread(x) for x in a.imgs] if a.imgs else load_from_hf(a.hf, a.split, a.count, a.offset, a.shuffle)
    if any(im is None for im in imgs):
        sys.exit("Falha ao ler alguma imagem.")

    st = make_stitcher(a.modo, a.warper, a.conf)
    status, pano = st.stitch(imgs)
    if status != cv2.STITCHER_OK:
        sys.exit(f"falhou com status {status}")
    cv2.imwrite(a.out, pano)
    print(f"ok: {a.out}")
    return 0
```

---

## `modules/datasets/loaders.py` — Interface comum de carregamento

```python
# modules/datasets/loaders.py
from dataclasses import dataclass
from typing import List, Literal, Optional
import cv2

SourceType = Literal["local","hf"]

@dataclass
class LoadSpec:
    source: SourceType
    paths: Optional[List[str]] = None
    hf_name: Optional[str] = None
    split: str = "train"
    count: int = 5
    offset: int = 0
    shuffle: bool = False

def pil_to_bgr(pil_img):
    import numpy as np, cv2
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def load_images(spec: LoadSpec):
    if spec.source == "local":
        imgs = [cv2.imread(p) for p in (spec.paths or [])]
        return [im for im in imgs if im is not None]
    elif spec.source == "hf":
        from datasets import load_dataset
        ds = load_dataset(spec.hf_name, split=spec.split)
        if spec.shuffle:
            ds = ds.shuffle(seed=42)
        end = min(len(ds), spec.offset + spec.count)
        imgs = []
        for rec in ds.select(range(spec.offset, end)):
            img = rec.get("image") or rec.get("rgb") or rec[list(rec.keys())[0]]
            try:
                imgs.append(pil_to_bgr(img))
            except Exception:
                if isinstance(img, str):
                    im = cv2.imread(img)
                    if im is not None:
                        imgs.append(im)
        return imgs
    else:
        raise ValueError("source inválida")
```

---

## `modules/benchmarks/run_bench.py` — Varredura automática (CSV + resumo)

```python
# modules/benchmarks/run_bench.py
import argparse, csv, os, time, subprocess

def cli_main():
    p = argparse.ArgumentParser(prog="benchmarks.run_bench")
    p.add_argument("--hf", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--counts", nargs="+", type=int, default=[16,24,32])
    p.add_argument("--offset_start", type=int, default=0)
    p.add_argument("--offset_end", type=int, default=1500)
    p.add_argument("--offset_step", type=int, default=50)
    p.add_argument("--modo", choices=["panorama","scans"], default="scans")
    p.add_argument("--warper", default="auto")
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--outdir", default="reports")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "bench.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["offset","count","status","seconds","outfile"])
        for count in args.counts:
            for off in range(args.offset_start, args.offset_end+1, args.offset_step):
                out = os.path.join(args.outdir, f"pano_{off}_{count}.jpg")
                cmd = ["python","main.py","classic.stitcher","--hf",args.hf,"--split",args.split,
                       "--count",str(count),"--offset",str(off),"--modo",args.modo,
                       "--warper",args.warper,"--conf",str(args.conf),"--out",out]
                t0=time.time()
                try:
                    subprocess.check_output(cmd, stderr=subprocess.STDOUT)
                    status="OK"
                except subprocess.CalledProcessError as e:
                    status=e.output.decode(errors="ignore").strip()[-3:]
                dt=time.time()-t0
                w.writerow([off,count,status,f"{dt:.2f}",out])
    print(f"salvo: {csv_path}")
    return 0

if __name__ == "__main__":
    cli_main()
```

---

## `requirements.txt` (mínimo)

```text
opencv-contrib-python==4.10.*
datasets
pillow
huggingface_hub[hf_xet]
```

> Com isso, `main.py` vira seu **script principal** do TCC, e o `classic.stitcher` (ex-1\_stitcher\_basico) é só um dos módulos. Quando quiser, eu crio também o esqueleto de `classic/features_homography.py` e dos módulos **deep** (LoFTR, SP+SG) com flags consistentes para você plugar os pesos depois.
