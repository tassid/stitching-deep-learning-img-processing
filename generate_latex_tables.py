import pandas as pd
import numpy as np

# Ler dados
df = pd.read_csv('results/scan_20251027_192717/resultados_consolidados.csv', encoding='utf-8-sig')
df_success = df[df['status'] == 'sucesso'].copy()

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

# Calcular score composto
df_grouped['psnr_norm'] = (df_grouped['psnr_db'] - df_grouped['psnr_db'].min()) / (df_grouped['psnr_db'].max() - df_grouped['psnr_db'].min())
df_grouped['ssim_norm'] = (df_grouped['ssim'] - df_grouped['ssim'].min()) / (df_grouped['ssim'].max() - df_grouped['ssim'].min())
df_grouped['ghosting_norm'] = 1 - (df_grouped['ghosting'] - df_grouped['ghosting'].min()) / (df_grouped['ghosting'].max() - df_grouped['ghosting'].min())
df_grouped['seam_norm'] = 1 - (df_grouped['seam_energy'] - df_grouped['seam_energy'].min()) / (df_grouped['seam_energy'].max() - df_grouped['seam_energy'].min())

df_grouped['score'] = (
    0.30 * df_grouped['psnr_norm'] +
    0.30 * df_grouped['ssim_norm'] +
    0.25 * df_grouped['ghosting_norm'] +
    0.15 * df_grouped['seam_norm']
)

df_grouped = df_grouped.sort_values('score', ascending=False)

# ===== TABELA 1: TOP 5 MÉTODOS =====
print('% ===== TABELA 1: TOP 5 MÉTODOS =====')
print('\\begin{table}[htb]')
print('\\centering')
print('\\caption{Ranking dos 5 melhores métodos de fusão de imagens}')
print('\\label{tab:top5}')
print('\\small')
print('\\begin{tabular}{clcccc}')
print('\\hline')
print('\\textbf{Rank} & \\textbf{Método} & \\textbf{Score} & \\textbf{PSNR (dB)} & \\textbf{SSIM} & \\textbf{Ghosting} \\\\ \\hline')

top5 = df_grouped.head(5)
for idx, (metodo, row) in enumerate(top5.iterrows(), 1):
    label_short = metodo[1].replace('Clássico • ', '').replace('Deep • ', '')
    print(f'{idx} & {label_short} & {row["score"]:.4f} & {row["psnr_db"]:.2f} & {row["ssim"]:.4f} & {row["ghosting"]:.4f} \\\\')

print('\\hline')
print('\\end{tabular}')
print('\\fonte{Autoria própria (2025).}')
print('\\end{table}')
print('\n\n')

# ===== TABELA 2: CLÁSSICOS VS DEEP LEARNING =====
print('% ===== TABELA 2: CLÁSSICOS VS DEEP LEARNING =====')
df_success['categoria'] = df_success['metodo'].apply(lambda x: 'Deep Learning' if x.startswith('deep-') else 'Clássico')
comp = df_success.groupby('categoria').agg({
    'psnr_db': ['mean', 'count'],
    'ssim': 'mean',
    'ghosting': 'mean',
    'tempo_segundos': 'mean'
}).round(4)

print('\\begin{table}[htb]')
print('\\centering')
print('\\caption{Comparação entre métodos clássicos e deep learning}')
print('\\label{tab:classic-vs-deep}')
print('\\begin{tabular}{lcccc}')
print('\\hline')
print('\\textbf{Categoria} & \\textbf{PSNR Médio (dB)} & \\textbf{SSIM Médio} & \\textbf{Ghosting} & \\textbf{Tempo (s)} \\\\ \\hline')

for cat in ['Clássico', 'Deep Learning']:
    if cat in comp.index:
        psnr = comp.loc[cat, ('psnr_db', 'mean')]
        ssim = comp.loc[cat, ('ssim', 'mean')]
        ghosting = comp.loc[cat, ('ghosting', 'mean')]
        tempo = comp.loc[cat, ('tempo_segundos', 'mean')]
        count = int(comp.loc[cat, ('psnr_db', 'count')])
        print(f'{cat} (n={count}) & {psnr:.2f} & {ssim:.4f} & {ghosting:.4f} & {tempo:.2f} \\\\')

print('\\hline')
print('\\end{tabular}')
print('\\fonte{Autoria própria (2025).}')
print('\\end{table}')
print('\n\n')

# ===== TABELA 3: RANKING POR PSNR (TOP 10) =====
print('% ===== TABELA 3: RANKING POR PSNR (TOP 10) =====')
print('\\begin{table}[htb]')
print('\\centering')
print('\\caption{Ranking dos métodos por PSNR (Top 10)}')
print('\\label{tab:psnr-ranking}')
print('\\small')
print('\\begin{tabular}{clcc}')
print('\\hline')
print('\\textbf{Pos.} & \\textbf{Método} & \\textbf{PSNR (dB)} & \\textbf{Categoria} \\\\ \\hline')

df_by_psnr = df_grouped.sort_values('psnr_db', ascending=False).head(10)
for idx, (metodo, row) in enumerate(df_by_psnr.iterrows(), 1):
    label_short = metodo[1].replace('Clássico • ', '').replace('Deep • ', '')
    categoria = 'Deep Learning' if metodo[0].startswith('deep-') else 'Clássico'
    print(f'{idx} & {label_short} & {row["psnr_db"]:.2f} & {categoria} \\\\')

print('\\hline')
print('\\end{tabular}')
print('\\fonte{Autoria própria (2025).}')
print('\\end{table}')
print('\n\n')

# ===== TABELA 4: RANKING POR SSIM (TOP 10) =====
print('% ===== TABELA 4: RANKING POR SSIM (TOP 10) =====')
print('\\begin{table}[htb]')
print('\\centering')
print('\\caption{Ranking dos métodos por SSIM (Top 10)}')
print('\\label{tab:ssim-ranking}')
print('\\small')
print('\\begin{tabular}{clcc}')
print('\\hline')
print('\\textbf{Pos.} & \\textbf{Método} & \\textbf{SSIM} & \\textbf{Interpretação} \\\\ \\hline')

df_by_ssim = df_grouped.sort_values('ssim', ascending=False).head(10)
for idx, (metodo, row) in enumerate(df_by_ssim.iterrows(), 1):
    label_short = metodo[1].replace('Clássico • ', '').replace('Deep • ', '')
    interpretacao = 'Excelente' if row['ssim'] >= 0.28 else 'Bom'
    print(f'{idx} & {label_short} & {row["ssim"]:.4f} & {interpretacao} \\\\')

print('\\hline')
print('\\end{tabular}')
print('\\fonte{Autoria própria (2025).}')
print('\\end{table}')
print('\n\n')

# ===== TABELA 5: RANKING POR GHOSTING (TOP 10) =====
print('% ===== TABELA 5: RANKING POR GHOSTING - MENOR É MELHOR (TOP 10) =====')
print('\\begin{table}[htb]')
print('\\centering')
print('\\caption{Ranking dos métodos por Ghosting Score (menor é melhor)}')
print('\\label{tab:ghosting-ranking}')
print('\\small')
print('\\begin{tabular}{clcc}')
print('\\hline')
print('\\textbf{Pos.} & \\textbf{Método} & \\textbf{Ghosting} & \\textbf{Interpretação} \\\\ \\hline')

df_by_ghosting = df_grouped.sort_values('ghosting', ascending=True).head(10)
for idx, (metodo, row) in enumerate(df_by_ghosting.iterrows(), 1):
    label_short = metodo[1].replace('Clássico • ', '').replace('Deep • ', '')
    interpretacao = 'Baixo' if row['ghosting'] < 0.94 else 'Moderado'
    print(f'{idx} & {label_short} & {row["ghosting"]:.4f} & {interpretacao} \\\\')

print('\\hline')
print('\\end{tabular}')
print('\\fonte{Autoria própria (2025).}')
print('\\end{table}')
print('\n\n')

# ===== TABELA 6: DETALHES DOS TOP 5 =====
print('% ===== TABELA 6: DETALHES COMPLETOS DOS TOP 5 =====')
print('\\begin{table}[htb]')
print('\\centering')
print('\\caption{Métricas detalhadas dos 5 melhores métodos}')
print('\\label{tab:top5-detalhado}')
print('\\tiny')
print('\\begin{tabular}{clcccccc}')
print('\\hline')
print('\\textbf{Rank} & \\textbf{Método} & \\textbf{PSNR} & \\textbf{SSIM} & \\textbf{Ghosting} & \\textbf{Seam} & \\textbf{Entropia} & \\textbf{Tempo (s)} \\\\ \\hline')

for idx, (metodo, row) in enumerate(top5.iterrows(), 1):
    label_short = metodo[1].replace('Clássico • ', '').replace('Deep • ', '')[:30]
    print(f'{idx} & {label_short} & {row["psnr_db"]:.2f} & {row["ssim"]:.4f} & {row["ghosting"]:.4f} & {row["seam_energy"]:.2f} & {row["entropia"]:.2f} & {row["tempo_segundos"]:.2f} \\\\')

print('\\hline')
print('\\end{tabular}')
print('\\fonte{Autoria própria (2025).}')
print('\\end{table}')

print('\n\n=== Tabelas LaTeX geradas com sucesso! ===')
