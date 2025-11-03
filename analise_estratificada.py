"""
Analise Estratificada por Tipo de Cenario
==========================================
Este script categoriza os datasets por tipo de desafio e compara
performance de metodos classicos vs deep learning em cada categoria.

Autor: Claude Code
Data: 2025-10-28
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuracao
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class AnalisadorEstrategico:
    def __init__(self, csv_path):
        """Carrega dados e categoriza por tipo de cenario"""
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.df = self.df[self.df['status'] == 'sucesso'].copy()

        # Adicionar categoria (classico vs deep)
        self.df['categoria'] = self.df['metodo'].apply(
            lambda x: 'Classico' if 'classic' in x else 'Deep Learning'
        )

        # Definir categorias de desafio baseado na literatura
        self.categorias_cenario = {
            'Baixa Textura': ['ski_left'],
            'Estruturas Repetitivas': [
                'rental_office_1',
                'graffiti_building_1',
                'no_contractor_parking_1',
                'stop_sign_1'
            ],
            'Paralaxe Significativa': [
                'cars_1',
                'cup_tree_1',
                'traffic_cones_1',
                'plant_1'
            ],
            'Variacao Iluminacao': [
                'coke_190',
                'building_dubai'
            ],
            'Superficies Curvas': [
                'graffiti_car_1'
            ],
            'Multiplas Imagens': [
                'building_dubai',
                'P2_front_of_window'
            ],
            'Cenarios Simples': [
                'biker_mural_1',
                'theory_left'
            ]
        }

        # Mapear cada dataset para sua(s) categoria(s)
        self.df['tipo_cenario'] = self.df['pasta'].apply(self._categorizar_dataset)

        print(f"[OK] Dados carregados: {len(self.df)} execucoes")
        print(f"  - Classicos: {sum(self.df['categoria'] == 'Classico')}")
        print(f"  - Deep Learning: {sum(self.df['categoria'] == 'Deep Learning')}")

    def _categorizar_dataset(self, pasta):
        """Retorna a categoria principal de um dataset"""
        for categoria, datasets in self.categorias_cenario.items():
            if pasta in datasets:
                return categoria
        return 'Outro'

    def analise_por_cenario(self):
        """Compara classico vs deep learning em cada tipo de cenario"""
        print("\n" + "="*80)
        print("ANALISE ESTRATIFICADA POR TIPO DE CENARIO")
        print("="*80)

        resultados = []

        for tipo_cenario in self.categorias_cenario.keys():
            print(f"\n### {tipo_cenario}")
            print("-" * 60)

            # Filtrar dados deste cenario
            dados_cenario = self.df[self.df['tipo_cenario'] == tipo_cenario]

            if len(dados_cenario) == 0:
                print("  [!] Sem dados para este cenario")
                continue

            classico = dados_cenario[dados_cenario['categoria'] == 'Classico']
            deep = dados_cenario[dados_cenario['categoria'] == 'Deep Learning']

            if len(classico) == 0 or len(deep) == 0:
                print(f"  [!] Dados insuficientes (Classico: {len(classico)}, Deep: {len(deep)})")
                continue

            print(f"  Datasets: {self.categorias_cenario[tipo_cenario]}")
            print(f"  N (Classico): {len(classico)}")
            print(f"  N (Deep Learning): {len(deep)}")

            # Comparar metricas principais
            metricas = ['psnr_db', 'ssim', 'ghosting', 'tempo_segundos']

            for metrica in metricas:
                media_classico = classico[metrica].mean()
                media_deep = deep[metrica].mean()

                # Mann-Whitney U test (nao-parametrico)
                try:
                    u_stat, p_value = mannwhitneyu(
                        classico[metrica].dropna(),
                        deep[metrica].dropna(),
                        alternative='two-sided'
                    )

                    # Calcular Cohen's d
                    mean_diff = media_classico - media_deep
                    pooled_std = np.sqrt(
                        (classico[metrica].std()**2 + deep[metrica].std()**2) / 2
                    )
                    cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0

                    # Determinar vencedor
                    if p_value < 0.05:
                        if metrica in ['psnr_db', 'ssim']:  # Maior eh melhor
                            vencedor = 'Deep' if media_deep > media_classico else 'Classico'
                        else:  # Menor eh melhor (ghosting, tempo)
                            vencedor = 'Deep' if media_deep < media_classico else 'Classico'
                        sig_str = f"[OK] {vencedor} superior"
                    else:
                        vencedor = 'Empate'
                        sig_str = "[X] Sem diferenca"

                    resultados.append({
                        'Tipo_Cenario': tipo_cenario,
                        'Metrica': metrica,
                        'Media_Classico': media_classico,
                        'Media_Deep': media_deep,
                        'Diferenca': mean_diff,
                        'p_value': p_value,
                        'Cohens_d': cohens_d,
                        'Significativo': 'Sim' if p_value < 0.05 else 'Nao',
                        'Vencedor': vencedor
                    })

                    print(f"    {metrica:18s}: Classico={media_classico:.3f}, Deep={media_deep:.3f}, "
                          f"p={p_value:.4f}, d={cohens_d:.3f} -> {sig_str}")

                except Exception as e:
                    print(f"    {metrica:18s}: [!] Erro: {e}")

        return pd.DataFrame(resultados)

    def resumo_vencedores(self, df_resultados):
        """Cria resumo de quantas vezes cada abordagem venceu"""
        print("\n" + "="*80)
        print("RESUMO: VENCEDORES POR TIPO DE CENARIO")
        print("="*80)

        # Contar vitorias
        vitorias = df_resultados[df_resultados['Significativo'] == 'Sim'].groupby(
            ['Tipo_Cenario', 'Vencedor']
        ).size().unstack(fill_value=0)

        print("\nVitorias por cenario (apenas diferencas significativas):")
        print(vitorias.to_string())

        # Resumo geral
        print("\n### Resumo Geral:")
        total_testes = len(df_resultados)
        testes_sig = len(df_resultados[df_resultados['Significativo'] == 'Sim'])

        vitorias_classico = len(df_resultados[
            (df_resultados['Significativo'] == 'Sim') &
            (df_resultados['Vencedor'] == 'Classico')
        ])

        vitorias_deep = len(df_resultados[
            (df_resultados['Significativo'] == 'Sim') &
            (df_resultados['Vencedor'] == 'Deep')
        ])

        print(f"  Total de testes: {total_testes}")
        print(f"  Testes com diferenca significativa: {testes_sig} ({testes_sig/total_testes*100:.1f}%)")
        print(f"  Vitorias Classico: {vitorias_classico}")
        print(f"  Vitorias Deep Learning: {vitorias_deep}")
        print(f"  Empates: {total_testes - testes_sig}")

    def visualizar_por_cenario(self, df_resultados, output_dir='results/analise_estratificada'):
        """Cria visualizacoes dos resultados por cenario"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Heatmap de p-values
        pivot_pvalue = df_resultados.pivot_table(
            values='p_value',
            index='Tipo_Cenario',
            columns='Metrica'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_pvalue, annot=True, fmt='.3f', cmap='RdYlGn_r',
                    vmin=0, vmax=0.1, cbar_kws={'label': 'p-value'})
        plt.title('P-values: Classico vs Deep Learning por Cenario\n(Verde = diferenca significativa)')
        plt.tight_layout()
        plt.savefig(output_path / 'heatmap_pvalues_por_cenario.png', dpi=300)
        print(f"[OK] Salvo: heatmap_pvalues_por_cenario.png")
        plt.close()

        # 2. Heatmap de Cohen's d (tamanho do efeito)
        pivot_cohens = df_resultados.pivot_table(
            values='Cohens_d',
            index='Tipo_Cenario',
            columns='Metrica'
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_cohens, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, cbar_kws={'label': "Cohen's d"})
        plt.title("Tamanho do Efeito (Cohen's d) por Cenario\n(Positivo = Classico melhor, Negativo = Deep melhor)")
        plt.tight_layout()
        plt.savefig(output_path / 'heatmap_cohens_d_por_cenario.png', dpi=300)
        print(f"[OK] Salvo: heatmap_cohens_d_por_cenario.png")
        plt.close()

        # 3. Bar plot de medias por cenario e metrica
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metricas = ['psnr_db', 'ssim', 'ghosting', 'tempo_segundos']

        for idx, metrica in enumerate(metricas):
            ax = axes[idx // 2, idx % 2]

            dados_metrica = df_resultados[df_resultados['Metrica'] == metrica]

            x = np.arange(len(dados_metrica['Tipo_Cenario'].unique()))
            width = 0.35

            classico_vals = dados_metrica.groupby('Tipo_Cenario')['Media_Classico'].first()
            deep_vals = dados_metrica.groupby('Tipo_Cenario')['Media_Deep'].first()

            ax.bar(x - width/2, classico_vals, width, label='Classico', alpha=0.8)
            ax.bar(x + width/2, deep_vals, width, label='Deep Learning', alpha=0.8)

            ax.set_xlabel('Tipo de Cenario')
            ax.set_ylabel(metrica)
            ax.set_title(f'{metrica} por Tipo de Cenario')
            ax.set_xticks(x)
            ax.set_xticklabels(classico_vals.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'barplot_metricas_por_cenario.png', dpi=300)
        print(f"[OK] Salvo: barplot_metricas_por_cenario.png")
        plt.close()

        # 4. Scatter plot: PSNR vs Tempo colorido por cenario
        fig, ax = plt.subplots(figsize=(12, 8))

        for tipo_cenario in self.df['tipo_cenario'].unique():
            if tipo_cenario == 'Outro':
                continue

            dados_cenario = self.df[self.df['tipo_cenario'] == tipo_cenario]

            ax.scatter(dados_cenario[dados_cenario['categoria'] == 'Classico']['tempo_segundos'],
                      dados_cenario[dados_cenario['categoria'] == 'Classico']['psnr_db'],
                      label=f'{tipo_cenario} (Classico)', alpha=0.6, s=80, marker='o')

            ax.scatter(dados_cenario[dados_cenario['categoria'] == 'Deep Learning']['tempo_segundos'],
                      dados_cenario[dados_cenario['categoria'] == 'Deep Learning']['psnr_db'],
                      label=f'{tipo_cenario} (Deep)', alpha=0.6, s=80, marker='^')

        ax.set_xlabel('Tempo de Processamento (s)')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Qualidade vs Velocidade por Tipo de Cenario')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'scatter_psnr_tempo_por_cenario.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Salvo: scatter_psnr_tempo_por_cenario.png")
        plt.close()

    def gerar_relatorio_latex(self, df_resultados, output_dir='results/analise_estratificada'):
        """Gera secao LaTeX pronta para incluir no TCC"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        latex_file = output_path / 'secao_analise_estratificada.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write("\\section{Análise Estratificada por Tipo de Cenário}\n\n")

            f.write("Para compreender melhor as condições sob as quais cada abordagem ")
            f.write("(clássica ou deep learning) apresenta desempenho superior, os 15 conjuntos ")
            f.write("de imagens foram categorizados de acordo com seus principais desafios técnicos. ")
            f.write("Esta análise estratificada permite identificar cenários específicos onde cada ")
            f.write("metodologia demonstra vantagens competitivas.\n\n")

            f.write("\\subsection{Categorização dos Cenários}\n\n")
            f.write("Os conjuntos de dados foram classificados nas seguintes categorias:\n\n")
            f.write("\\begin{itemize}\n")
            for categoria, datasets in self.categorias_cenario.items():
                f.write(f"    \\item \\textbf{{{categoria}}}: {', '.join(datasets)}\n")
            f.write("\\end{itemize}\n\n")

            # Tabela de resultados
            f.write("\\subsection{Resultados Comparativos por Categoria}\n\n")
            f.write("A Tabela~\\ref{tab:estratificado} apresenta os resultados dos testes estatísticos ")
            f.write("comparando métodos clássicos e deep learning em cada categoria de cenário.\n\n")

            f.write("\\begin{table}[htb]\n")
            f.write("\\centering\n")
            f.write("\\caption{Análise estratificada: Clássico vs Deep Learning por tipo de cenário}\n")
            f.write("\\label{tab:estratificado}\n")
            f.write("\\tiny\n")
            f.write("\\begin{tabular}{llrrrcp{3cm}}\n")
            f.write("\\hline\n")
            f.write("\\textbf{Cenário} & \\textbf{Métrica} & \\textbf{Clássico} & \\textbf{Deep} & ")
            f.write("\\textbf{p-value} & \\textbf{Sig.} & \\textbf{Interpretação} \\\\ \\hline\n")

            for _, row in df_resultados.iterrows():
                sig_symbol = "$\\checkmark$" if row['Significativo'] == 'Sim' else "$\\times$"

                # Interpretacao
                if row['Significativo'] == 'Sim':
                    interp = f"{row['Vencedor']} superior"
                else:
                    interp = "Sem diferença"

                f.write(f"{row['Tipo_Cenario']} & {row['Metrica']} & "
                       f"{row['Media_Classico']:.3f} & {row['Media_Deep']:.3f} & "
                       f"{row['p_value']:.4f} & {sig_symbol} & {interp} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\fonte{Autoria própria (2025).}\n")
            f.write("\\end{table}\n\n")

            # Discussao
            f.write("\\subsection{Discussão dos Resultados Estratificados}\n\n")

            # Contar vitorias
            vitorias_classico = len(df_resultados[
                (df_resultados['Significativo'] == 'Sim') &
                (df_resultados['Vencedor'] == 'Classico')
            ])
            vitorias_deep = len(df_resultados[
                (df_resultados['Significativo'] == 'Sim') &
                (df_resultados['Vencedor'] == 'Deep')
            ])
            total_sig = len(df_resultados[df_resultados['Significativo'] == 'Sim'])
            total_testes = len(df_resultados)

            f.write(f"Dos {total_testes} testes realizados, {total_sig} apresentaram diferenças ")
            f.write(f"estatisticamente significativas (p<0,05), correspondendo a {total_sig/total_testes*100:.1f}\\% ")
            f.write(f"dos casos. Destes, métodos clássicos foram superiores em {vitorias_classico} casos, ")
            f.write(f"enquanto deep learning superou em {vitorias_deep} casos.\n\n")

            # Insights por categoria
            f.write("\\textbf{Principais Observações:}\n\n")
            f.write("\\begin{enumerate}\n")

            # Analisar cada categoria
            for tipo_cenario in df_resultados['Tipo_Cenario'].unique():
                dados_cat = df_resultados[df_resultados['Tipo_Cenario'] == tipo_cenario]
                vit_deep_cat = len(dados_cat[
                    (dados_cat['Significativo'] == 'Sim') &
                    (dados_cat['Vencedor'] == 'Deep')
                ])
                vit_class_cat = len(dados_cat[
                    (dados_cat['Significativo'] == 'Sim') &
                    (dados_cat['Vencedor'] == 'Classico')
                ])

                if vit_deep_cat > vit_class_cat:
                    f.write(f"    \\item \\textbf{{{tipo_cenario}}}: Deep learning demonstrou superioridade ")
                    f.write(f"nesta categoria, vencendo em {vit_deep_cat} de {len(dados_cat)} métricas.\n")
                elif vit_class_cat > vit_deep_cat:
                    f.write(f"    \\item \\textbf{{{tipo_cenario}}}: Métodos clássicos foram superiores, ")
                    f.write(f"vencendo em {vit_class_cat} de {len(dados_cat)} métricas.\n")
                else:
                    f.write(f"    \\item \\textbf{{{tipo_cenario}}}: Ambas as abordagens apresentaram ")
                    f.write(f"desempenho equivalente, sem diferenças estatisticamente significativas.\n")

            f.write("\\end{enumerate}\n\n")

            # Figuras
            f.write("A Figura~\\ref{fig:heatmap_pvalues} apresenta um mapa de calor dos p-values obtidos ")
            f.write("para cada combinação de cenário e métrica.\n\n")

            f.write("\\begin{figure}[htb]\n")
            f.write("\\centering\n")
            f.write("\\includegraphics[width=0.8\\textwidth]{results/analise_estratificada/heatmap_pvalues_por_cenario.png}\n")
            f.write("\\caption{P-values dos testes Mann-Whitney U por tipo de cenário e métrica}\n")
            f.write("\\label{fig:heatmap_pvalues}\n")
            f.write("\\fonte{Autoria própria (2025).}\n")
            f.write("\\end{figure}\n\n")

        print(f"[OK] Secao LaTeX salva em: {latex_file}")
        print("    Copie este arquivo para o capitulo de Resultados do seu TCC!")

def main():
    csv_path = 'results/scan_20251027_192717/resultados_consolidados.csv'

    print("="*80)
    print("ANALISE ESTRATIFICADA POR TIPO DE CENARIO")
    print("="*80)

    analisador = AnalisadorEstrategico(csv_path)

    # Executar analise
    df_resultados = analisador.analise_por_cenario()

    # Salvar resultados
    output_dir = Path('results/analise_estratificada')
    output_dir.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(output_dir / 'resultados_estratificados.csv', index=False)
    print(f"\n[OK] Resultados salvos em: {output_dir / 'resultados_estratificados.csv'}")

    # Resumo
    analisador.resumo_vencedores(df_resultados)

    # Visualizacoes
    print("\n" + "="*80)
    print("GERANDO VISUALIZACOES...")
    print("="*80)
    analisador.visualizar_por_cenario(df_resultados)

    # Relatorio LaTeX
    print("\n" + "="*80)
    print("GERANDO SECAO LATEX...")
    print("="*80)
    analisador.gerar_relatorio_latex(df_resultados)

    print("\n" + "="*80)
    print("[OK] ANALISE ESTRATIFICADA CONCLUIDA COM SUCESSO!")
    print("="*80)
    print("\nArquivos gerados em: results/analise_estratificada/")
    print("  - resultados_estratificados.csv")
    print("  - heatmap_pvalues_por_cenario.png")
    print("  - heatmap_cohens_d_por_cenario.png")
    print("  - barplot_metricas_por_cenario.png")
    print("  - scatter_psnr_tempo_por_cenario.png")
    print("  - secao_analise_estratificada.tex")

if __name__ == "__main__":
    main()
