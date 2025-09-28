# Sistema de Métricas (Projeto TCC)

Este documento descreve o **sistema de métricas** implementado para avaliar mosaicos/panoramas.

## Arquivos
- `modules/classic/metrics.py` — versão básica (SSIM/MSE/PSNR, alinhamento, máscara)
- `modules/classic/metrics_plus.py` — versão avançada (acrescenta **seam_energy**, **ghosting**, **DeltaE**, **overlap_ratio**, tempo)
- `modules/utils/metrics_core.py` — núcleo com funções reutilizáveis

## Métricas calculadas
- **SSIM (cinza)** — similaridade estrutural [0..1] (↑ melhor)
- **MSE / PSNR (cor)** — erro quadrático médio e razão sinal/ruído (↓/↑ melhor)
- **overlap_ratio** — fração de pixels válidos em comum (↑ melhor garante comparação justa)
- **seam_energy** — energia média do gradiente de |A−B| no overlap (↓ melhor = menos costura visível)
- **ghosting** — 1 − IoU de bordas (Canny) entre A e B no overlap (↓ melhor = bordas coincidem)
- **deltaE_mean (Lab)** — diferença média de cor/iluminação (↓ melhor)
- **deltaL_mean, L_ref, L_cmp** — estatísticas fotométricas em L*

## Modos de uso
### Referência + candidatos
```powershell
python -m modules.classic.metrics_plus --ref outs/pano_ref.jpg --imgs outs/pano_A.jpg outs/pano_B.jpg --mask-nonzero --align resize --outdir reports_metrics_plus
```

### Diretório (todas as combinações)
```powershell
python -m modules.classic.metrics_plus --dir .\outs --align resize --mask-nonzero --outdir reports_metrics_plus
```

## Decisões de projeto
- **Área válida**: comparação restrita ao **overlap** para evitar viés de fundo preto.
- **Alinhamento**: `resize` por padrão; `pad`/`crop` disponíveis para análises de sensibilidade.
- **SSIM cinza**: robusto a pequenas variações cromáticas; PSNR/MSE cobrem aspecto radiométrico.

## Extensões sugeridas
- Registrar **inliers** e **RMSE** de reprojeção nos scripts de alinhamento e salvar `.stats.json` ao lado do `.jpg`.
- Agregação no `benchmarks/` para unir **métricas** + **tempos** + **inliers** num único CSV.
