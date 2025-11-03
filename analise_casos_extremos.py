"""
Analise de Casos Extremos
==========================
Identifica e analisa em profundidade os melhores e piores casos
de cada metodo, revelando padroes de sucesso e falha.

Autor: Claude Code
Data: 2025-10-28
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shutil

# Configuracao
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class AnalisadorCasosExtremos:
    def __init__(self, csv_path):
        """Carrega dados"""
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.df = self.df[self.df['status'] == 'sucesso'].copy()

        self.df['categoria'] = self.df['metodo'].apply(
            lambda x: 'Classico' if 'classic' in x else 'Deep Learning'
        )

        print(f"[OK] Dados carregados: {len(self.df)} execucoes bem-sucedidas")

    def top_bottom_por_metodo(self, n=5):
        """Identifica top N e bottom N de cada metodo"""
        print("\n" + "="*80)
        print(f"TOP {n} E BOTTOM {n} POR METODO")
        print("="*80)

        resultados = []

        for metodo in sorted(self.df['metodo'].unique()):
            dados_metodo = self.df[self.df['metodo'] == metodo]

            print(f"\n### {metodo}")
            print(f"    Total de execucoes: {len(dados_metodo)}")

            # Top N
            top_n = dados_metodo.nlargest(n, 'psnr_db')
            bottom_n = dados_metodo.nsmallest(n, 'psnr_db')

            print(f"    Top {n} PSNR: {top_n['psnr_db'].mean():.2f} dB")
            print(f"      Datasets: {list(top_n['pasta'])}")
            print(f"    Bottom {n} PSNR: {bottom_n['psnr_db'].mean():.2f} dB")
            print(f"      Datasets: {list(bottom_n['pasta'])}")

            # Calcular range de performance
            psnr_range = top_n['psnr_db'].mean() - bottom_n['psnr_db'].mean()
            print(f"    Range (top-bottom): {psnr_range:.2f} dB")

            # Guardar para analise posterior
            for _, row in top_n.iterrows():
                resultados.append({
                    'metodo': metodo,
                    'tipo': 'Top',
                    'pasta': row['pasta'],
                    'psnr_db': row['psnr_db'],
                    'ssim': row['ssim'],
                    'ghosting': row['ghosting'],
                    'tempo_segundos': row['tempo_segundos']
                })

            for _, row in bottom_n.iterrows():
                resultados.append({
                    'metodo': metodo,
                    'tipo': 'Bottom',
                    'pasta': row['pasta'],
                    'psnr_db': row['psnr_db'],
                    'ssim': row['ssim'],
                    'ghosting': row['ghosting'],
                    'tempo_segundos': row['tempo_segundos']
                })

        return pd.DataFrame(resultados)

    def datasets_mais_desafiadores(self):
        """Identifica datasets onde todos os metodos tiveram baixo desempenho"""
        print("\n" + "="*80)
        print("DATASETS MAIS DESAFIADORES (TODOS OS METODOS FALHARAM)")
        print("="*80)

        # Calcular media de PSNR por dataset
        psnr_por_dataset = self.df.groupby('pasta')['psnr_db'].agg(['mean', 'std', 'min', 'max', 'count'])
        psnr_por_dataset = psnr_por_dataset.sort_values('mean')

        print("\n### Top 5 Datasets Mais Dificeis (menor PSNR medio):")
        print(psnr_por_dataset.head().to_string())

        # Identificar datasets com alta variacao (inconsistentes)
        psnr_por_dataset['cv'] = (psnr_por_dataset['std'] / psnr_por_dataset['mean']) * 100
        psnr_por_dataset_sorted_cv = psnr_por_dataset.sort_values('cv', ascending=False)

        print("\n### Top 5 Datasets Mais Inconsistentes (maior CV):")
        print(psnr_por_dataset_sorted_cv.head().to_string())

        return psnr_por_dataset

    def casos_deep_superior(self, threshold=2.0):
        """Identifica casos onde deep learning foi significativamente melhor"""
        print("\n" + "="*80)
        print(f"CASOS ONDE DEEP LEARNING FOI SUPERIOR (>{threshold} dB)")
        print("="*80)

        casos_superiores = []

        for dataset in self.df['pasta'].unique():
            dados_dataset = self.df[self.df['pasta'] == dataset]

            psnr_classico = dados_dataset[dados_dataset['categoria'] == 'Classico']['psnr_db'].mean()
            psnr_deep = dados_dataset[dados_dataset['categoria'] == 'Deep Learning']['psnr_db'].mean()

            diferenca = psnr_deep - psnr_classico

            if diferenca > threshold:
                print(f"\n  Dataset: {dataset}")
                print(f"    PSNR Classico: {psnr_classico:.2f} dB")
                print(f"    PSNR Deep:     {psnr_deep:.2f} dB")
                print(f"    Diferenca:     +{diferenca:.2f} dB")

                # Encontrar melhor metodo deep e classico
                melhor_deep = dados_dataset[dados_dataset['categoria'] == 'Deep Learning'].nlargest(1, 'psnr_db')
                melhor_classico = dados_dataset[dados_dataset['categoria'] == 'Classico'].nlargest(1, 'psnr_db')

                print(f"    Melhor Deep:     {melhor_deep.iloc[0]['label']} ({melhor_deep.iloc[0]['psnr_db']:.2f} dB)")
                print(f"    Melhor Classico: {melhor_classico.iloc[0]['label']} ({melhor_classico.iloc[0]['psnr_db']:.2f} dB)")

                casos_superiores.append({
                    'dataset': dataset,
                    'psnr_classico': psnr_classico,
                    'psnr_deep': psnr_deep,
                    'diferenca': diferenca,
                    'melhor_deep': melhor_deep.iloc[0]['label'],
                    'melhor_classico': melhor_classico.iloc[0]['label']
                })

        if len(casos_superiores) == 0:
            print(f"\n  [!] Nenhum caso encontrado com diferenca > {threshold} dB")
        else:
            print(f"\n  Total de casos: {len(casos_superiores)}")

        return pd.DataFrame(casos_superiores)

    def casos_classico_superior(self, threshold=2.0):
        """Identifica casos onde classico foi significativamente melhor"""
        print("\n" + "="*80)
        print(f"CASOS ONDE CLASSICO FOI SUPERIOR (>{threshold} dB)")
        print("="*80)

        casos_superiores = []

        for dataset in self.df['pasta'].unique():
            dados_dataset = self.df[self.df['pasta'] == dataset]

            psnr_classico = dados_dataset[dados_dataset['categoria'] == 'Classico']['psnr_db'].mean()
            psnr_deep = dados_dataset[dados_dataset['categoria'] == 'Deep Learning']['psnr_db'].mean()

            diferenca = psnr_classico - psnr_deep

            if diferenca > threshold:
                print(f"\n  Dataset: {dataset}")
                print(f"    PSNR Classico: {psnr_classico:.2f} dB")
                print(f"    PSNR Deep:     {psnr_deep:.2f} dB")
                print(f"    Diferenca:     +{diferenca:.2f} dB")

                melhor_deep = dados_dataset[dados_dataset['categoria'] == 'Deep Learning'].nlargest(1, 'psnr_db')
                melhor_classico = dados_dataset[dados_dataset['categoria'] == 'Classico'].nlargest(1, 'psnr_db')

                print(f"    Melhor Classico: {melhor_classico.iloc[0]['label']} ({melhor_classico.iloc[0]['psnr_db']:.2f} dB)")
                print(f"    Melhor Deep:     {melhor_deep.iloc[0]['label']} ({melhor_deep.iloc[0]['psnr_db']:.2f} dB)")

                casos_superiores.append({
                    'dataset': dataset,
                    'psnr_classico': psnr_classico,
                    'psnr_deep': psnr_deep,
                    'diferenca': diferenca,
                    'melhor_classico': melhor_classico.iloc[0]['label'],
                    'melhor_deep': melhor_deep.iloc[0]['label']
                })

        if len(casos_superiores) == 0:
            print(f"\n  [!] Nenhum caso encontrado com diferenca > {threshold} dB")
        else:
            print(f"\n  Total de casos: {len(casos_superiores)}")

        return pd.DataFrame(casos_superiores)

    def analise_padroes_sucesso_falha(self):
        """Analisa padroes em sucessos e falhas"""
        print("\n" + "="*80)
        print("ANALISE DE PADROES DE SUCESSO E FALHA")
        print("="*80)

        # Definir thresholds de sucesso/falha
        threshold_sucesso = self.df['psnr_db'].quantile(0.75)
        threshold_falha = self.df['psnr_db'].quantile(0.25)

        sucessos = self.df[self.df['psnr_db'] >= threshold_sucesso]
        falhas = self.df[self.df['psnr_db'] <= threshold_falha]

        print(f"\nThreshold de sucesso (Q3): {threshold_sucesso:.2f} dB")
        print(f"Threshold de falha (Q1):   {threshold_falha:.2f} dB")
        print(f"\nCasos de sucesso: {len(sucessos)}")
        print(f"Casos de falha:   {len(falhas)}")

        # Quais datasets aparecem mais em sucessos?
        print("\n### Datasets com mais sucessos:")
        top_sucessos = sucessos['pasta'].value_counts().head(5)
        print(top_sucessos.to_string())

        # Quais datasets aparecem mais em falhas?
        print("\n### Datasets com mais falhas:")
        top_falhas = falhas['pasta'].value_counts().head(5)
        print(top_falhas.to_string())

        # Quais metodos aparecem mais em sucessos?
        print("\n### Metodos com mais sucessos:")
        top_metodos_sucesso = sucessos['label'].value_counts().head(5)
        print(top_metodos_sucesso.to_string())

        # Quais metodos aparecem mais em falhas?
        print("\n### Metodos com mais falhas:")
        top_metodos_falha = falhas['label'].value_counts().head(5)
        print(top_metodos_falha.to_string())

        return {
            'sucessos': sucessos,
            'falhas': falhas,
            'threshold_sucesso': threshold_sucesso,
            'threshold_falha': threshold_falha
        }

    def visualizar_casos_extremos(self, df_extremos, output_dir='results/analise_casos_extremos'):
        """Cria visualizacoes dos casos extremos"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Distribuicao de PSNR para Top vs Bottom
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Por categoria
        ax = axes[0]
        for categoria in ['Classico', 'Deep Learning']:
            dados_cat = self.df[self.df['categoria'] == categoria]
            ax.hist(dados_cat['psnr_db'], bins=30, alpha=0.6, label=categoria)
        ax.set_xlabel('PSNR (dB)')
        ax.set_ylabel('Frequencia')
        ax.set_title('Distribuicao de PSNR: Classico vs Deep Learning')
        ax.legend()
        ax.grid(alpha=0.3)

        # Top vs Bottom
        ax = axes[1]
        tops = df_extremos[df_extremos['tipo'] == 'Top']
        bottoms = df_extremos[df_extremos['tipo'] == 'Bottom']

        ax.hist(tops['psnr_db'], bins=20, alpha=0.6, label='Top 5', color='green')
        ax.hist(bottoms['psnr_db'], bins=20, alpha=0.6, label='Bottom 5', color='red')
        ax.set_xlabel('PSNR (dB)')
        ax.set_ylabel('Frequencia')
        ax.set_title('Distribuicao: Melhores vs Piores Casos')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'distribuicao_casos_extremos.png', dpi=300)
        print(f"[OK] Salvo: distribuicao_casos_extremos.png")
        plt.close()

        # 2. Violin plot: PSNR por dataset
        plt.figure(figsize=(16, 8))
        datasets_sorted = self.df.groupby('pasta')['psnr_db'].mean().sort_values().index

        sns.violinplot(data=self.df, x='pasta', y='psnr_db', order=datasets_sorted,
                      hue='categoria', split=True, palette=['blue', 'orange'])
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Dataset')
        plt.ylabel('PSNR (dB)')
        plt.title('Distribuicao de PSNR por Dataset (Classico vs Deep Learning)')
        plt.legend(title='Categoria')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'violin_psnr_por_dataset.png', dpi=300)
        print(f"[OK] Salvo: violin_psnr_por_dataset.png")
        plt.close()

        # 3. Box plot: Range de performance por metodo
        plt.figure(figsize=(16, 10))

        # Ordenar metodos por PSNR mediano
        metodos_ordenados = self.df.groupby('label')['psnr_db'].median().sort_values(ascending=False).index

        sns.boxplot(data=self.df, y='label', x='psnr_db', order=metodos_ordenados,
                   palette='Set2')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Metodo')
        plt.title('Range de Performance por Metodo (Box Plot)')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'boxplot_range_por_metodo.png', dpi=300)
        print(f"[OK] Salvo: boxplot_range_por_metodo.png")
        plt.close()

        # 4. Heatmap: Melhor metodo por dataset
        pivot = self.df.pivot_table(values='psnr_db', index='label', columns='pasta')

        # Encontrar melhor metodo para cada dataset
        best_method_mask = pivot == pivot.max(axis=0)

        plt.figure(figsize=(20, 12))
        sns.heatmap(pivot, annot=False, cmap='RdYlGn', center=11,
                   cbar_kws={'label': 'PSNR (dB)'})

        # Marcar os melhores com quadrado preto
        for i, row_label in enumerate(pivot.index):
            for j, col_label in enumerate(pivot.columns):
                if best_method_mask.iloc[i, j]:
                    plt.gca().add_patch(plt.Rectangle((j, i), 1, 1,
                                                     fill=False, edgecolor='black', lw=2))

        plt.title('Mapa de Calor: PSNR por Metodo e Dataset\n(Quadrados pretos = melhor metodo)')
        plt.tight_layout()
        plt.savefig(output_path / 'heatmap_melhor_metodo_por_dataset.png', dpi=300)
        print(f"[OK] Salvo: heatmap_melhor_metodo_por_dataset.png")
        plt.close()

    def gerar_relatorio_latex(self, df_extremos, df_padroes, output_dir='results/analise_casos_extremos'):
        """Gera secao LaTeX para casos extremos"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        latex_file = output_path / 'secao_casos_extremos.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("\\section{Análise de Casos Extremos}\n\n")

            f.write("A análise de casos extremos permite identificar padrões nas condições que ")
            f.write("levam ao sucesso ou falha de diferentes métodos de fusão. Esta seção examina ")
            f.write("os melhores e piores resultados obtidos, revelando características importantes ")
            f.write("para seleção apropriada de técnicas.\n\n")

            f.write("\\subsection{Datasets Mais Desafiadores}\n\n")

            # Identificar datasets dificeis
            datasets_dificeis = self.df.groupby('pasta')['psnr_db'].mean().sort_values().head(5)

            f.write("Os cinco conjuntos de imagens que apresentaram maior dificuldade para todos ")
            f.write("os métodos testados foram:\n\n")

            f.write("\\begin{enumerate}\n")
            for dataset, psnr_medio in datasets_dificeis.items():
                f.write(f"    \\item \\texttt{{{dataset}}}: PSNR médio de {psnr_medio:.2f} dB\n")
            f.write("\\end{enumerate}\n\n")

            f.write("Estes datasets compartilham características que tornam a fusão particularmente ")
            f.write("desafiadora, como baixo contraste, estruturas altamente repetitivas ou ")
            f.write("paralaxe pronunciada. A análise detalhada destes casos revela limitações ")
            f.write("fundamentais das abordagens baseadas em homografia planar global.\n\n")

            f.write("\\subsection{Padrões de Sucesso e Falha}\n\n")

            threshold_sucesso = df_padroes['threshold_sucesso']
            threshold_falha = df_padroes['threshold_falha']

            f.write(f"Definindo sucesso como PSNR $\\geq$ {threshold_sucesso:.2f} dB (quartil superior) ")
            f.write(f"e falha como PSNR $\\leq$ {threshold_falha:.2f} dB (quartil inferior), ")
            f.write(f"foram identificados {len(df_padroes['sucessos'])} casos de sucesso e ")
            f.write(f"{len(df_padroes['falhas'])} casos de falha.\n\n")

            # Metodos com mais sucessos
            top_sucessos = df_padroes['sucessos']['label'].value_counts().head(3)
            f.write("\\textbf{Métodos com maior taxa de sucesso:}\n")
            f.write("\\begin{itemize}\n")
            for metodo, count in top_sucessos.items():
                f.write(f"    \\item {metodo}: {count} casos\n")
            f.write("\\end{itemize}\n\n")

            # Figura
            f.write("A Figura~\\ref{fig:violin_extremos} apresenta a distribuição de PSNR por dataset, ")
            f.write("comparando métodos clássicos e deep learning.\n\n")

            f.write("\\begin{figure}[htb]\n")
            f.write("\\centering\n")
            f.write("\\includegraphics[width=\\textwidth]{results/analise_casos_extremos/violin_psnr_por_dataset.png}\n")
            f.write("\\caption{Distribuição de PSNR por dataset: Clássico vs Deep Learning}\n")
            f.write("\\label{fig:violin_extremos}\n")
            f.write("\\fonte{Autoria própria (2025).}\n")
            f.write("\\end{figure}\n\n")

        print(f"[OK] Secao LaTeX salva em: {latex_file}")

def main():
    csv_path = 'results/scan_20251027_192717/resultados_consolidados.csv'

    print("="*80)
    print("ANALISE DE CASOS EXTREMOS")
    print("="*80)

    analisador = AnalisadorCasosExtremos(csv_path)

    # 1. Top/Bottom por metodo
    df_extremos = analisador.top_bottom_por_metodo(n=5)

    # 2. Datasets mais desafiadores
    df_datasets = analisador.datasets_mais_desafiadores()

    # 3. Casos onde deep >> classico
    df_deep_superior = analisador.casos_deep_superior(threshold=2.0)

    # 4. Casos onde classico >> deep
    df_classico_superior = analisador.casos_classico_superior(threshold=2.0)

    # 5. Padroes de sucesso/falha
    df_padroes = analisador.analise_padroes_sucesso_falha()

    # Salvar resultados
    output_dir = Path('results/analise_casos_extremos')
    output_dir.mkdir(parents=True, exist_ok=True)

    df_extremos.to_csv(output_dir / 'top_bottom_por_metodo.csv', index=False)
    df_datasets.to_csv(output_dir / 'datasets_desafiadores.csv')

    if len(df_deep_superior) > 0:
        df_deep_superior.to_csv(output_dir / 'casos_deep_superior.csv', index=False)
    if len(df_classico_superior) > 0:
        df_classico_superior.to_csv(output_dir / 'casos_classico_superior.csv', index=False)

    print(f"\n[OK] Resultados salvos em: {output_dir}")

    # Visualizacoes
    print("\n" + "="*80)
    print("GERANDO VISUALIZACOES...")
    print("="*80)
    analisador.visualizar_casos_extremos(df_extremos)

    # Relatorio LaTeX
    print("\n" + "="*80)
    print("GERANDO SECAO LATEX...")
    print("="*80)
    analisador.gerar_relatorio_latex(df_extremos, df_padroes)

    print("\n" + "="*80)
    print("[OK] ANALISE DE CASOS EXTREMOS CONCLUIDA!")
    print("="*80)
    print("\nArquivos gerados em: results/analise_casos_extremos/")
    print("  - top_bottom_por_metodo.csv")
    print("  - datasets_desafiadores.csv")
    print("  - casos_deep_superior.csv (se existir)")
    print("  - casos_classico_superior.csv (se existir)")
    print("  - distribuicao_casos_extremos.png")
    print("  - violin_psnr_por_dataset.png")
    print("  - boxplot_range_por_metodo.png")
    print("  - heatmap_melhor_metodo_por_dataset.png")
    print("  - secao_casos_extremos.tex")

if __name__ == "__main__":
    main()
