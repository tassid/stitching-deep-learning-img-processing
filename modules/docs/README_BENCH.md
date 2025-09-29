# Benchmarks — Execução + Métricas

Este agregador executa **múltiplos módulos**, mede **tempo** por job e calcula **métricas** (SSIM/MSE/PSNR + ghosting + seam_energy + DeltaE + overlap) entre as saídas.

## Arquivo de jobs (JSON)
Veja `jobs_example.json` e `jobs_deep.json` como modelos. Cada job informa:
- `name`: rótulo curto
- `cmd`: lista com o comando (use `{out}` no lugar do caminho de saída)
- `outfile`: nome do arquivo resultante

## Exemplos (Clássicos)
```powershell
python benchmarks/run_bench_and_score.py --jobs benchmarks/jobs_example.json --outdir outs --align resize --mask-nonzero
```

## Exemplos (Deep)
### Rodar todos os módulos deep e pontuar pairwise
```powershell
python benchmarks/run_bench_and_score.py --jobs benchmarks/jobs_deep.json --outdir outs_deep --align resize --mask-nonzero
```

### Comparar resultados deep vs um panorama de referência
```powershell
python benchmarks/run_bench_and_score.py --jobs benchmarks/jobs_deep.json --outdir outs_deep
python benchmarks/run_bench_and_score.py --score-only --dir outs_deep --ref outs_deep/deep_loftr.jpg --align resize --mask-nonzero
```
