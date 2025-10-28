import os
import shutil
from pathlib import Path

# Diretórios
source_dir = Path('results/scan_20251027_192717')
tcc_figs_dir = Path('TCC_TASSIANE_2025_NOVOTEMPLATE/figuras/resultados')

# Criar diretório se não existir
tcc_figs_dir.mkdir(parents=True, exist_ok=True)

# Top 5 métodos (baseado na análise)
top5_methods = [
    {'rank': 1, 'id': 'classic-opencv_stitcher', 'label': 'OpenCV Stitcher'},
    {'rank': 2, 'id': 'classic-1_stitcher_basico', 'label': 'Stitcher Basico'},
    {'rank': 3, 'id': 'classic-6_seams_exposure_multiband', 'label': 'Seams Exposure Multiband'},
    {'rank': 4, 'id': 'classic-8_blend_laplacian', 'label': 'Blend Laplacian'},
    {'rank': 5, 'id': 'classic-9_exposure_comp_only', 'label': 'Exposure Comp Only'}
]

# Copiar imagens comparison do before_after
print('Copiando imagens de comparação...')
copied = 0
for method in top5_methods:
    method_id = method['id']
    # Procurar imagens comparison no before_after
    pattern = f'*_{method_id}_comparison.jpg'
    comparison_imgs = list((source_dir / 'before_after').glob(pattern))

    for img in comparison_imgs[:5]:  # Limitar a 5 exemplos por método
        dataset_name = img.stem.replace(f'_{method_id}_comparison', '')
        new_name = f'rank{method["rank"]:02d}_{dataset_name}_{method_id}_comparison.jpg'
        dest = tcc_figs_dir / new_name

        if not dest.exists():
            shutil.copy2(img, dest)
            print(f'  Copiado: {new_name}')
            copied += 1

print(f'\nTotal de imagens copiadas: {copied}')

# Criar um relatório de imagens disponíveis
print('\n=== Imagens disponíveis por método ===')
for method in top5_methods:
    method_id = method['id']
    pattern = f'*_{method_id}_comparison.jpg'
    imgs = list((source_dir / 'before_after').glob(pattern))
    print(f'\nRank {method["rank"]} - {method["label"]}: {len(imgs)} imagens')
    for img in imgs[:3]:
        dataset = img.stem.replace(f'_{method_id}_comparison', '')
        print(f'  - {dataset}')
