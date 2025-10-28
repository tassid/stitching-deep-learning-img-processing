import pandas as pd
import numpy as np

# Ler o CSV
df = pd.read_csv('results/scan_20251027_192717/resultados_consolidados.csv', encoding='utf-8-sig')

# Filtrar apenas sucessos
df_success = df[df['status'] == 'sucesso'].copy()

print('=== ANÁLISE DOS MELHORES MÉTODOS ===\n')
print(f'Total de execuções bem-sucedidas: {len(df_success)}')
print(f'Total de métodos testados: {df_success["metodo"].nunique()}')
print(f'Total de datasets testados: {df_success["pasta"].nunique()}\n')

# Converter colunas para numérico
metrics = ['psnr_db', 'ssim', 'mse', 'ghosting', 'seam_energy', 'deltaE_mean', 'entropia', 'nitidez', 'tempo_segundos']
for col in metrics:
    if col in df_success.columns:
        df_success[col] = pd.to_numeric(df_success[col], errors='coerce')

# Agrupar por método
df_grouped = df_success.groupby(['metodo', 'label']).agg({
    'psnr_db': 'mean',
    'ssim': 'mean',
    'ghosting': 'mean',
    'seam_energy': 'mean',
    'tempo_segundos': 'mean',
    'deltaE_mean': 'mean',
    'entropia': 'mean',
    'nitidez': 'mean',
    'mse': 'mean'
}).round(4)

# Calcular score ponderado (quanto maior melhor)
# Normalizar métricas e criar score composto
df_grouped['psnr_norm'] = (df_grouped['psnr_db'] - df_grouped['psnr_db'].min()) / (df_grouped['psnr_db'].max() - df_grouped['psnr_db'].min())
df_grouped['ssim_norm'] = (df_grouped['ssim'] - df_grouped['ssim'].min()) / (df_grouped['ssim'].max() - df_grouped['ssim'].min())
df_grouped['ghosting_norm'] = 1 - (df_grouped['ghosting'] - df_grouped['ghosting'].min()) / (df_grouped['ghosting'].max() - df_grouped['ghosting'].min())
df_grouped['seam_norm'] = 1 - (df_grouped['seam_energy'] - df_grouped['seam_energy'].min()) / (df_grouped['seam_energy'].max() - df_grouped['seam_energy'].min())

# Score composto: 30% PSNR, 30% SSIM, 25% Ghosting, 15% Seam Energy
df_grouped['score'] = (
    0.30 * df_grouped['psnr_norm'] +
    0.30 * df_grouped['ssim_norm'] +
    0.25 * df_grouped['ghosting_norm'] +
    0.15 * df_grouped['seam_norm']
)

# Ordenar por score
df_grouped = df_grouped.sort_values('score', ascending=False)

print('\n=== TOP 10 MÉTODOS (por score composto) ===\n')
top10 = df_grouped.head(10)[['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos', 'score']]
print(top10.to_string())

print('\n\n=== TOP 5 MÉTODOS DETALHADO ===\n')
top5 = df_grouped.head(5)
for idx, (metodo, row) in enumerate(top5.iterrows(), 1):
    print(f'\n{idx}. {metodo[1]}')
    print(f'   Método ID: {metodo[0]}')
    print(f'   Score: {row["score"]:.4f}')
    print(f'   PSNR: {row["psnr_db"]:.2f} dB')
    print(f'   SSIM: {row["ssim"]:.4f}')
    print(f'   Ghosting: {row["ghosting"]:.4f}')
    print(f'   Seam Energy: {row["seam_energy"]:.2f}')
    print(f'   Tempo médio: {row["tempo_segundos"]:.2f}s')
    print(f'   DeltaE: {row["deltaE_mean"]:.2f}')
    print(f'   Entropia: {row["entropia"]:.2f}')
    print(f'   Nitidez: {row["nitidez"]:.2f}')

print('\n\n=== COMPARAÇÃO CLÁSSICOS vs DEEP LEARNING ===\n')
df_success['categoria'] = df_success['metodo'].apply(lambda x: 'Deep Learning' if x.startswith('deep-') else 'Clássico')
comp = df_success.groupby('categoria').agg({
    'psnr_db': 'mean',
    'ssim': 'mean',
    'ghosting': 'mean',
    'tempo_segundos': 'mean',
    'mse': 'mean'
}).round(4)
print(comp.to_string())

# Salvar top 5 em arquivo
top5_methods = df_grouped.head(5).reset_index()
top5_methods.to_csv('results/scan_20251027_192717/top5_methods.csv', index=False, encoding='utf-8-sig')
print('\n\nTop 5 salvos em: results/scan_20251027_192717/top5_methods.csv')
