"""
Visualizacoes Avancadas
=======================
Cria graficos profissionais para apresentacao dos resultados.

Autor: Claude Code
Data: 2025-10-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Configuracao
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

class VisualizadorAvancado:
    def __init__(self, csv_path):
        """Carrega dados"""
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.df = self.df[self.df['status'] == 'sucesso'].copy()

        self.df['categoria'] = self.df['metodo'].apply(
            lambda x: 'Classico' if 'classic' in x else 'Deep Learning'
        )

        print(f"[OK] Dados carregados: {len(self.df)} execucoes")

    def boxplots_comparativos(self, output_dir):
        """Box plots para todas as metricas principais"""
        print("\n[1/7] Gerando box plots comparativos...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos', 'entropia']
        titulos = ['PSNR (dB)', 'SSIM', 'Ghosting Score', 'Seam Energy', 'Tempo (s)', 'Entropia']

        for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
            ax = axes[idx // 3, idx % 3]

            sns.boxplot(data=self.df, x='categoria', y=metrica, ax=ax,
                       palette=['#3498db', '#e74c3c'])

            # Adicionar media como linha
            for i, categoria in enumerate(['Classico', 'Deep Learning']):
                media = self.df[self.df['categoria'] == categoria][metrica].mean()
                ax.hlines(media, i-0.4, i+0.4, colors='red', linestyles='--',
                         linewidth=2, label='Media' if i == 0 else '')

            ax.set_xlabel('Categoria', fontweight='bold')
            ax.set_ylabel(titulo, fontweight='bold')
            ax.set_title(f'{titulo} por Categoria', fontweight='bold', fontsize=12)
            ax.grid(axis='y', alpha=0.3)

            if idx == 0:
                ax.legend()

        plt.suptitle('Analise Comparativa: Metodos Classicos vs Deep Learning',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(output_dir / 'boxplots_comparativos.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    [OK] boxplots_comparativos.png")

    def heatmap_performance(self, output_dir):
        """Heatmap: metodos x datasets"""
        print("\n[2/7] Gerando heatmap de performance...")

        # Pivot table: linhas=metodos, colunas=datasets, valores=PSNR
        pivot = self.df.pivot_table(values='psnr_db', index='label', columns='pasta', aggfunc='mean')

        # Ordenar por PSNR medio
        pivot['media'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('media', ascending=False)
        pivot = pivot.drop('media', axis=1)

        # Criar figura grande
        plt.figure(figsize=(20, 14))

        # Heatmap com anotacoes
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=11, vmin=7, vmax=15,
                   cbar_kws={'label': 'PSNR (dB)', 'shrink': 0.8},
                   linewidths=0.5, linecolor='white')

        plt.title('Mapa de Calor: PSNR por Metodo e Dataset', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Metodo', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_dir / 'heatmap_psnr_metodo_dataset.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    [OK] heatmap_psnr_metodo_dataset.png")

    def scatter_3d_qualidade_velocidade(self, output_dir):
        """Scatter 3D: PSNR x SSIM x Tempo"""
        print("\n[3/7] Gerando scatter 3D qualidade vs velocidade...")

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plotar por categoria
        for categoria, cor, marker in [('Classico', '#3498db', 'o'),
                                       ('Deep Learning', '#e74c3c', '^')]:
            dados = self.df[self.df['categoria'] == categoria]

            ax.scatter(dados['psnr_db'], dados['ssim'], dados['tempo_segundos'],
                      c=cor, marker=marker, s=100, alpha=0.6,
                      label=categoria, edgecolors='black', linewidth=0.5)

        ax.set_xlabel('PSNR (dB)', fontweight='bold', labelpad=10)
        ax.set_ylabel('SSIM', fontweight='bold', labelpad=10)
        ax.set_zlabel('Tempo (s)', fontweight='bold', labelpad=10)
        ax.set_title('Espaco 3D: Qualidade vs Velocidade', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'scatter_3d_qualidade_velocidade.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    [OK] scatter_3d_qualidade_velocidade.png")

    def histogramas_distribuicao(self, output_dir):
        """Histogramas de distribuicao das metricas"""
        print("\n[4/7] Gerando histogramas de distribuicao...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos', 'entropia']
        titulos = ['PSNR (dB)', 'SSIM', 'Ghosting Score', 'Seam Energy', 'Tempo (s)', 'Entropia']

        for idx, (metrica, titulo) in enumerate(zip(metricas, titulos)):
            ax = axes[idx // 3, idx % 3]

            # Histogramas sobrepostos
            for categoria, cor, alpha in [('Classico', '#3498db', 0.6),
                                         ('Deep Learning', '#e74c3c', 0.6)]:
                dados = self.df[self.df['categoria'] == categoria][metrica]

                ax.hist(dados, bins=25, alpha=alpha, label=categoria,
                       color=cor, edgecolor='black', linewidth=0.5)

                # Adicionar linha de media
                media = dados.mean()
                ax.axvline(media, color=cor, linestyle='--', linewidth=2,
                          label=f'Media {categoria}')

            ax.set_xlabel(titulo, fontweight='bold')
            ax.set_ylabel('Frequencia', fontweight='bold')
            ax.set_title(f'Distribuicao: {titulo}', fontweight='bold', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Distribuicoes das Metricas por Categoria',
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(output_dir / 'histogramas_distribuicao.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    [OK] histogramas_distribuicao.png")

    def matriz_correlacao(self, output_dir):
        """Matriz de correlacao entre metricas"""
        print("\n[5/7] Gerando matriz de correlacao...")

        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos', 'entropia']
        labels = ['PSNR', 'SSIM', 'Ghosting', 'Seam Energy', 'Tempo', 'Entropia']

        # Calcular correlacao de Pearson
        corr_matrix = self.df[metricas].corr(method='pearson')

        # Criar figura
        fig, ax = plt.subplots(figsize=(10, 8))

        # Heatmap triangular
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                   cmap='coolwarm', center=0, vmin=-1, vmax=1,
                   square=True, linewidths=1, cbar_kws={'shrink': 0.8, 'label': 'Correlacao de Pearson'},
                   xticklabels=labels, yticklabels=labels, ax=ax)

        plt.title('Matriz de Correlacao entre Metricas', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(output_dir / 'matriz_correlacao.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    [OK] matriz_correlacao.png")

    def pareto_chart(self, output_dir):
        """Grafico de Pareto dos metodos por score"""
        print("\n[6/7] Gerando grafico de Pareto...")

        # Calcular score normalizado para cada metodo
        # Score = media(PSNR_norm + SSIM_norm - Ghosting_norm - Tempo_norm)

        # Normalizar metricas [0, 1]
        df_norm = self.df.copy()
        df_norm['psnr_norm'] = (df_norm['psnr_db'] - df_norm['psnr_db'].min()) / \
                                (df_norm['psnr_db'].max() - df_norm['psnr_db'].min())
        df_norm['ssim_norm'] = (df_norm['ssim'] - df_norm['ssim'].min()) / \
                                (df_norm['ssim'].max() - df_norm['ssim'].min())
        df_norm['ghosting_norm'] = 1 - ((df_norm['ghosting'] - df_norm['ghosting'].min()) / \
                                         (df_norm['ghosting'].max() - df_norm['ghosting'].min()))
        df_norm['tempo_norm'] = 1 - ((df_norm['tempo_segundos'] - df_norm['tempo_segundos'].min()) / \
                                      (df_norm['tempo_segundos'].max() - df_norm['tempo_segundos'].min()))

        # Calcular score
        df_norm['score'] = (df_norm['psnr_norm'] + df_norm['ssim_norm'] +
                            df_norm['ghosting_norm'] + df_norm['tempo_norm']) / 4

        # Agrupar por metodo
        scores_por_metodo = df_norm.groupby('label')['score'].mean().sort_values(ascending=False)

        # Calcular percentual acumulado
        total = scores_por_metodo.sum()
        percentual_acumulado = (scores_por_metodo.cumsum() / total * 100)

        # Criar figura
        fig, ax1 = plt.subplots(figsize=(16, 10))

        # Barras
        x = np.arange(len(scores_por_metodo))
        bars = ax1.bar(x, scores_por_metodo, color='#3498db', alpha=0.7, edgecolor='black')

        # Destacar top 5
        for i in range(min(5, len(bars))):
            bars[i].set_color('#27ae60')

        ax1.set_xlabel('Metodo', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score Normalizado', fontsize=12, fontweight='bold', color='#3498db')
        ax1.set_xticks(x)
        ax1.set_xticklabels(scores_por_metodo.index, rotation=45, ha='right', fontsize=8)
        ax1.tick_params(axis='y', labelcolor='#3498db')
        ax1.grid(axis='y', alpha=0.3)

        # Linha de percentual acumulado
        ax2 = ax1.twinx()
        ax2.plot(x, percentual_acumulado, color='#e74c3c', marker='o',
                linewidth=2, markersize=4, label='% Acumulado')
        ax2.set_ylabel('Percentual Acumulado (%)', fontsize=12, fontweight='bold', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        ax2.set_ylim([0, 105])

        # Linha 80%
        ax2.axhline(80, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax2.text(len(x)-1, 82, '80%', fontsize=10, color='gray')

        plt.title('Grafico de Pareto: Ranking de Metodos por Score Composto\n(Verde = Top 5)',
                 fontsize=14, fontweight='bold', pad=15)
        fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
        plt.tight_layout()
        plt.savefig(output_dir / 'pareto_chart_metodos.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    [OK] pareto_chart_metodos.png")

    def radar_chart_top5(self, output_dir):
        """Radar chart comparando top 5 metodos"""
        print("\n[7/7] Gerando radar chart dos top 5...")

        # Calcular score e pegar top 5
        df_norm = self.df.copy()
        df_norm['psnr_norm'] = (df_norm['psnr_db'] - df_norm['psnr_db'].min()) / \
                                (df_norm['psnr_db'].max() - df_norm['psnr_db'].min())
        df_norm['ssim_norm'] = (df_norm['ssim'] - df_norm['ssim'].min()) / \
                                (df_norm['ssim'].max() - df_norm['ssim'].min())
        df_norm['ghosting_norm'] = 1 - ((df_norm['ghosting'] - df_norm['ghosting'].min()) / \
                                         (df_norm['ghosting'].max() - df_norm['ghosting'].min()))
        df_norm['seam_norm'] = 1 - ((df_norm['seam_energy'] - df_norm['seam_energy'].min()) / \
                                     (df_norm['seam_energy'].max() - df_norm['seam_energy'].min()))
        df_norm['tempo_norm'] = 1 - ((df_norm['tempo_segundos'] - df_norm['tempo_segundos'].min()) / \
                                      (df_norm['tempo_segundos'].max() - df_norm['tempo_segundos'].min()))

        df_norm['score'] = (df_norm['psnr_norm'] + df_norm['ssim_norm'] +
                            df_norm['ghosting_norm'] + df_norm['seam_norm'] +
                            df_norm['tempo_norm']) / 5

        top5_metodos = df_norm.groupby('label')['score'].mean().nlargest(5).index

        # Preparar dados para radar
        categorias = ['PSNR', 'SSIM', 'Ghosting', 'Seam', 'Velocidade']
        metricas_norm = ['psnr_norm', 'ssim_norm', 'ghosting_norm', 'seam_norm', 'tempo_norm']

        # Criar figura
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='polar')

        angles = np.linspace(0, 2 * np.pi, len(categorias), endpoint=False).tolist()
        angles += angles[:1]  # Fechar o circulo

        cores = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']

        for idx, metodo in enumerate(top5_metodos):
            dados_metodo = df_norm[df_norm['label'] == metodo]
            valores = [dados_metodo[m].mean() for m in metricas_norm]
            valores += valores[:1]  # Fechar o circulo

            ax.plot(angles, valores, 'o-', linewidth=2, label=metodo,
                   color=cores[idx], markersize=6)
            ax.fill(angles, valores, alpha=0.15, color=cores[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categorias, size=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
        ax.grid(True, alpha=0.3)

        plt.title('Radar Chart: Comparacao dos Top 5 Metodos\n(Valores normalizados [0,1])',
                 fontsize=14, fontweight='bold', pad=30)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.tight_layout()
        plt.savefig(output_dir / 'radar_chart_top5.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("    [OK] radar_chart_top5.png")

def main():
    csv_path = 'results/scan_20251027_192717/resultados_consolidados.csv'

    print("="*80)
    print("VISUALIZACOES AVANCADAS")
    print("="*80)

    visualizador = VisualizadorAvancado(csv_path)

    # Criar diretorio de saida
    output_dir = Path('results/visualizacoes_avancadas')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gerar todas as visualizacoes
    visualizador.boxplots_comparativos(output_dir)
    visualizador.heatmap_performance(output_dir)
    visualizador.scatter_3d_qualidade_velocidade(output_dir)
    visualizador.histogramas_distribuicao(output_dir)
    visualizador.matriz_correlacao(output_dir)
    visualizador.pareto_chart(output_dir)
    visualizador.radar_chart_top5(output_dir)

    print("\n" + "="*80)
    print("[OK] TODAS AS VISUALIZACOES FORAM GERADAS!")
    print("="*80)
    print(f"\nArquivos salvos em: {output_dir}/")
    print("\nLista completa:")
    print("  1. boxplots_comparativos.png")
    print("  2. heatmap_psnr_metodo_dataset.png")
    print("  3. scatter_3d_qualidade_velocidade.png")
    print("  4. histogramas_distribuicao.png")
    print("  5. matriz_correlacao.png")
    print("  6. pareto_chart_metodos.png")
    print("  7. radar_chart_top5.png")
    print("\nUse estas imagens no capitulo de Resultados do seu TCC!")

if __name__ == "__main__":
    main()
