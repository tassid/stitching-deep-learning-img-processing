"""
Script simples para visualizar metricas de forma organizada
"""

import sys
from pathlib import Path

print("="*80)
print("VISUALIZADOR DE METRICAS - RESULTS")
print("="*80)

# Verificar se pasta results existe
results_dir = Path("results")
if not results_dir.exists():
    print("\nERRO: Pasta 'results' nao encontrada!")
    print("Execute a GUI primeiro para gerar resultados.")
    sys.exit(1)

# Verificar arquivo de log
log_file = results_dir / "metrics_log.txt"
if not log_file.exists():
    print("\nAVISO: Arquivo metrics_log.txt nao encontrado!")
    print("Nenhuma execucao registrada ainda.")
    sys.exit(0)

# Ler log
with open(log_file, 'r', encoding='utf-8') as f:
    linhas = f.readlines()

print(f"\nTotal de execucoes registradas: {len(linhas)}")
print("="*80)

# Processar linhas
execucoes = []
for linha in linhas:
    linha = linha.strip()
    if not linha:
        continue

    # Parsear linha
    # Formato: TIMESTAMP [METODO] PSNR=XX.XXX SSIM=X.XXXX out=arquivo.jpg
    partes = linha.split()
    if len(partes) < 4:
        continue

    timestamp = partes[0]
    metodo = partes[1].strip('[]')

    psnr = None
    ssim = None
    arquivo = None

    for parte in partes[2:]:
        if parte.startswith("PSNR="):
            try:
                psnr = float(parte.split('=')[1])
            except:
                psnr = None
        elif parte.startswith("SSIM="):
            try:
                ssim = float(parte.split('=')[1])
            except:
                ssim = None
        elif parte.startswith("out="):
            arquivo = parte.split('=')[1]

    execucoes.append({
        'timestamp': timestamp,
        'metodo': metodo,
        'psnr': psnr,
        'ssim': ssim,
        'arquivo': arquivo
    })

# Mostrar ultimas 10 execucoes
print("\nULTIMAS 10 EXECUCOES:")
print("-"*80)
print(f"{'Data/Hora':<20} {'Metodo':<30} {'PSNR (dB)':<12} {'SSIM':<8}")
print("-"*80)

for exec in execucoes[-10:]:
    timestamp_fmt = exec['timestamp'][:8] + " " + exec['timestamp'][9:11] + ":" + exec['timestamp'][11:13]
    metodo = exec['metodo'][:28]

    if exec['psnr'] is not None and not (exec['psnr'] != exec['psnr']):  # Checar se nao e NaN
        psnr_str = f"{exec['psnr']:.3f}"
    else:
        psnr_str = "N/A"

    if exec['ssim'] is not None and not (exec['ssim'] != exec['ssim']):
        ssim_str = f"{exec['ssim']:.4f}"
    else:
        ssim_str = "N/A"

    print(f"{timestamp_fmt:<20} {metodo:<30} {psnr_str:<12} {ssim_str:<8}")

# Estatisticas
print("\n" + "="*80)
print("ESTATISTICAS")
print("="*80)

# Filtrar apenas execucoes validas (com metricas)
validas = [e for e in execucoes if e['psnr'] is not None and not (e['psnr'] != e['psnr'])]

if validas:
    print(f"\nExecutacoes com metricas validas: {len(validas)}/{len(execucoes)}")

    psnrs = [e['psnr'] for e in validas]
    ssims = [e['ssim'] for e in validas]

    print(f"\nPSNR:")
    print(f"  Melhor:  {max(psnrs):.3f} dB")
    print(f"  Pior:    {min(psnrs):.3f} dB")
    print(f"  Media:   {sum(psnrs)/len(psnrs):.3f} dB")

    print(f"\nSSIM:")
    print(f"  Melhor:  {max(ssims):.4f}")
    print(f"  Pior:    {min(ssims):.4f}")
    print(f"  Media:   {sum(ssims)/len(ssims):.4f}")

    # Melhor execucao por PSNR
    melhor_psnr = max(validas, key=lambda x: x['psnr'])
    print(f"\nMELHOR RESULTADO (PSNR):")
    print(f"  Metodo: {melhor_psnr['metodo']}")
    print(f"  PSNR:   {melhor_psnr['psnr']:.3f} dB")
    print(f"  SSIM:   {melhor_psnr['ssim']:.4f}")
    print(f"  Data:   {melhor_psnr['timestamp']}")
    print(f"  Arquivo: {melhor_psnr['arquivo']}")

    # Melhor execucao por SSIM
    melhor_ssim = max(validas, key=lambda x: x['ssim'])
    print(f"\nMELHOR RESULTADO (SSIM):")
    print(f"  Metodo: {melhor_ssim['metodo']}")
    print(f"  PSNR:   {melhor_ssim['psnr']:.3f} dB")
    print(f"  SSIM:   {melhor_ssim['ssim']:.4f}")
    print(f"  Data:   {melhor_ssim['timestamp']}")
    print(f"  Arquivo: {melhor_ssim['arquivo']}")
else:
    print("\nNenhuma execucao com metricas validas encontrada.")
    print("\nDICA: Para calcular metricas:")
    print("  1. Marque checkbox 'Calcular metricas vs referencia'")
    print("  2. Selecione imagem de ground truth")
    print("  3. Execute novamente")

print("\n" + "="*80)
print("ARQUIVOS DISPONIVEIS:")
print("="*80)

# Listar arquivos TXT
txt_files = sorted(results_dir.glob("pano_*.txt"))
print(f"\nArquivos TXT individuais: {len(txt_files)}")
if len(txt_files) <= 5:
    for f in txt_files:
        print(f"  - {f.name}")
else:
    for f in txt_files[:3]:
        print(f"  - {f.name}")
    print(f"  ... e mais {len(txt_files)-3} arquivos")

# Listar arquivos JPG
jpg_files = sorted(results_dir.glob("pano_*.jpg"))
print(f"\nImagens JPG geradas: {len(jpg_files)}")
if len(jpg_files) <= 5:
    for f in jpg_files:
        print(f"  - {f.name}")
else:
    for f in jpg_files[:3]:
        print(f"  - {f.name}")
    print(f"  ... e mais {len(jpg_files)-3} arquivos")

print("\n" + "="*80)
print("Para ver detalhes de uma execucao especifica:")
print("  Abra: results/pano_[metodo]_[timestamp].txt")
print("\nPara ver historico completo:")
print("  Abra: results/metrics_log.txt")
print("="*80)
