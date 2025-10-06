"""
Visualizador de resultados do pipeline de image stitching
Gera gráficos, tabelas LaTeX e análises estatísticas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional

# Tentar importar matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Backend não-interativo
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  matplotlib não encontrado. Gráficos serão desabilitados.")
    print("   Para gerar gráficos: pip install matplotlib")

# Tentar importar seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
    sns.set_style("whitegrid")
except ImportError:
    HAS_SEABORN = False


class VisualizadorResultados:
    """
    Classe para análise e visualização de resultados do pipeline
    """

    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: Caminho para o CSV com resultados
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.output_dir = self.csv_path.parent / "analises"
        self.output_dir.mkdir(exist_ok=True)

        print(f"✅ Resultados carregados: {len(self.df)} experimentos")
        print(f"💾 Análises serão salvas em: {self.output_dir}")

    def gerar_tabela_resumo_latex(self) -> str:
        """
        Gera tabela LaTeX resumida (médias por algoritmo)

        Returns:
            Código LaTeX da tabela
        """
        print("\n📝 Gerando tabela resumo LaTeX...")

        df_success = self.df[self.df['psnr_db'].notna()].copy()

        if len(df_success) == 0:
            print("⚠️  Nenhum experimento com métricas válidas")
            return "% Sem dados válidos"

        # Agrupar por algoritmo
        resumo = df_success.groupby('algoritmo').agg({
            'psnr_db': ['mean', 'std', 'count'],
            'ssim': ['mean', 'std'],
            'tempo_segundos': ['mean', 'std']
        }).round(4)

        # Gerar LaTeX
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Resumo das Métricas por Algoritmo}\n"
        latex += "\\label{tab:resumo_metricas}\n"
        latex += "\\begin{tabular}{lcccc}\n"
        latex += "\\hline\n"
        latex += "Algoritmo & PSNR (dB) & SSIM & Tempo (s) & Experimentos \\\\\n"
        latex += "\\hline\n"

        for alg in resumo.index:
            psnr_mean = resumo.loc[alg, ('psnr_db', 'mean')]
            psnr_std = resumo.loc[alg, ('psnr_db', 'std')]
            ssim_mean = resumo.loc[alg, ('ssim', 'mean')]
            ssim_std = resumo.loc[alg, ('ssim', 'std')]
            tempo_mean = resumo.loc[alg, ('tempo_segundos', 'mean')]
            count = int(resumo.loc[alg, ('psnr_db', 'count')])

            latex += f"{alg} & ${psnr_mean:.2f} \\pm {psnr_std:.2f}$ & "
            latex += f"${ssim_mean:.4f} \\pm {ssim_std:.4f}$ & "
            latex += f"${tempo_mean:.2f}$ & {count} \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        # Salvar
        output_file = self.output_dir / "tabela_resumo.tex"
        output_file.write_text(latex, encoding='utf-8')
        print(f"✅ Tabela resumo salva: {output_file}")

        return latex

    def gerar_tabela_detalhada_latex(self) -> str:
        """
        Gera tabela LaTeX detalhada (todos os experimentos)

        Returns:
            Código LaTeX da tabela
        """
        print("\n📝 Gerando tabela detalhada LaTeX...")

        df_success = self.df[self.df['psnr_db'].notna()].copy()

        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Resultados Detalhados por Conjunto e Algoritmo}\n"
        latex += "\\label{tab:resultados_detalhados}\n"
        latex += "\\small\n"
        latex += "\\begin{tabular}{llcccc}\n"
        latex += "\\hline\n"
        latex += "Conjunto & Algoritmo & PSNR (dB) & SSIM & Tempo (s) & Resolução \\\\\n"
        latex += "\\hline\n"

        for _, row in df_success.iterrows():
            conjunto = row['conjunto'].replace('_', '\\_')
            algoritmo = row['algoritmo'].replace('_', '\\_')
            psnr = row['psnr_db']
            ssim = row['ssim']
            tempo = row['tempo_segundos']
            resolucao = f"{int(row['largura'])}x{int(row['altura'])}" if pd.notna(row['largura']) else 'N/A'

            latex += f"{conjunto} & {algoritmo} & {psnr:.2f} & {ssim:.4f} & {tempo:.2f} & {resolucao} \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        # Salvar
        output_file = self.output_dir / "tabela_detalhada.tex"
        output_file.write_text(latex, encoding='utf-8')
        print(f"✅ Tabela detalhada salva: {output_file}")

        return latex

    def plotar_comparacao_psnr(self):
        """Plota gráfico de barras comparando PSNR por algoritmo"""
        if not HAS_MATPLOTLIB:
            print("⚠️  matplotlib não disponível. Pulando gráfico PSNR.")
            return

        print("\n📊 Gerando gráfico de PSNR...")

        df_success = self.df[self.df['psnr_db'].notna()].copy()

        if len(df_success) == 0:
            print("⚠️  Sem dados para plotar")
            return

        # Calcular médias
        medias = df_success.groupby('algoritmo')['psnr_db'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(medias)), medias.values)

        # Colorir barras
        colors = plt.cm.viridis(np.linspace(0, 1, len(medias)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Algoritmo', fontsize=12)
        ax.set_ylabel('PSNR Médio (dB)', fontsize=12)
        ax.set_title('Comparação de PSNR por Algoritmo', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(medias)))
        ax.set_xticklabels(medias.index, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

        # Adicionar valores nas barras
        for i, v in enumerate(medias.values):
            ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom')

        # Linha de referência (baseline do paper: ~19.49 dB)
        ax.axhline(y=19.49, color='r', linestyle='--', label='Baseline (paper)')
        ax.legend()

        plt.tight_layout()
        output_file = self.output_dir / "grafico_psnr.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Gráfico PSNR salvo: {output_file}")

    def plotar_comparacao_ssim(self):
        """Plota gráfico de barras comparando SSIM por algoritmo"""
        if not HAS_MATPLOTLIB:
            print("⚠️  matplotlib não disponível. Pulando gráfico SSIM.")
            return

        print("\n📊 Gerando gráfico de SSIM...")

        df_success = self.df[self.df['ssim'].notna()].copy()

        if len(df_success) == 0:
            print("⚠️  Sem dados para plotar")
            return

        medias = df_success.groupby('algoritmo')['ssim'].mean().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(medias)), medias.values)

        colors = plt.cm.plasma(np.linspace(0, 1, len(medias)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xlabel('Algoritmo', fontsize=12)
        ax.set_ylabel('SSIM Médio', fontsize=12)
        ax.set_title('Comparação de SSIM por Algoritmo', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(medias)))
        ax.set_xticklabels(medias.index, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)

        for i, v in enumerate(medias.values):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        output_file = self.output_dir / "grafico_ssim.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Gráfico SSIM salvo: {output_file}")

    def plotar_comparacao_tempo(self):
        """Plota gráfico comparando tempo de execução"""
        if not HAS_MATPLOTLIB:
            print("⚠️  matplotlib não disponível. Pulando gráfico de tempo.")
            return

        print("\n📊 Gerando gráfico de tempo...")

        df_success = self.df[self.df['tempo_segundos'].notna()].copy()

        if len(df_success) == 0:
            print("⚠️  Sem dados para plotar")
            return

        medias = df_success.groupby('algoritmo')['tempo_segundos'].mean().sort_values()

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(range(len(medias)), medias.values)

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(medias)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_ylabel('Algoritmo', fontsize=12)
        ax.set_xlabel('Tempo Médio (segundos)', fontsize=12)
        ax.set_title('Comparação de Tempo de Execução', fontsize=14, fontweight='bold')
        ax.set_yticks(range(len(medias)))
        ax.set_yticklabels(medias.index)
        ax.grid(axis='x', alpha=0.3)

        for i, v in enumerate(medias.values):
            ax.text(v + 0.1, i, f'{v:.2f}s', va='center')

        plt.tight_layout()
        output_file = self.output_dir / "grafico_tempo.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Gráfico de tempo salvo: {output_file}")

    def plotar_scatter_qualidade_tempo(self):
        """Plota scatter plot: qualidade (PSNR) vs tempo"""
        if not HAS_MATPLOTLIB:
            return

        print("\n📊 Gerando gráfico scatter qualidade vs tempo...")

        df_success = self.df[self.df['psnr_db'].notna() & self.df['tempo_segundos'].notna()].copy()

        if len(df_success) == 0:
            print("⚠️  Sem dados para plotar")
            return

        # Calcular médias por algoritmo
        medias = df_success.groupby('algoritmo').agg({
            'psnr_db': 'mean',
            'tempo_segundos': 'mean',
            'ssim': 'mean'
        })

        fig, ax = plt.subplots(figsize=(12, 8))

        # Scatter
        scatter = ax.scatter(
            medias['tempo_segundos'],
            medias['psnr_db'],
            s=200,
            c=medias['ssim'],
            cmap='viridis',
            alpha=0.6,
            edgecolors='black'
        )

        # Labels
        for alg, row in medias.iterrows():
            ax.annotate(
                alg,
                (row['tempo_segundos'], row['psnr_db']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )

        ax.set_xlabel('Tempo Médio (segundos)', fontsize=12)
        ax.set_ylabel('PSNR Médio (dB)', fontsize=12)
        ax.set_title('Trade-off: Qualidade vs Velocidade', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('SSIM Médio', fontsize=11)

        plt.tight_layout()
        output_file = self.output_dir / "grafico_scatter_trade_off.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Gráfico scatter salvo: {output_file}")

    def plotar_heatmap_psnr(self):
        """Plota heatmap: PSNR por (conjunto, algoritmo)"""
        if not HAS_MATPLOTLIB:
            return

        print("\n📊 Gerando heatmap de PSNR...")

        df_success = self.df[self.df['psnr_db'].notna()].copy()

        if len(df_success) == 0:
            print("⚠️  Sem dados para plotar")
            return

        # Pivot table
        pivot = df_success.pivot_table(
            values='psnr_db',
            index='conjunto',
            columns='algoritmo',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(14, 8))

        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')

        # Configurar eixos
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticklabels(pivot.index)

        # Adicionar valores
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    text = ax.text(j, i, f'{val:.1f}',
                                 ha="center", va="center", color="black", fontsize=9)

        ax.set_title('Heatmap: PSNR por Conjunto e Algoritmo', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PSNR (dB)', fontsize=11)

        plt.tight_layout()
        output_file = self.output_dir / "heatmap_psnr.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Heatmap salvo: {output_file}")

    def gerar_relatorio_completo(self):
        """Gera todos os gráficos e tabelas"""
        print("\n" + "="*80)
        print("📊 GERANDO RELATÓRIO COMPLETO")
        print("="*80)

        # Tabelas LaTeX
        self.gerar_tabela_resumo_latex()
        self.gerar_tabela_detalhada_latex()

        # Gráficos
        if HAS_MATPLOTLIB:
            self.plotar_comparacao_psnr()
            self.plotar_comparacao_ssim()
            self.plotar_comparacao_tempo()
            self.plotar_scatter_qualidade_tempo()
            self.plotar_heatmap_psnr()
        else:
            print("\n⚠️  matplotlib não disponível. Gráficos não gerados.")

        # Estatísticas textuais
        self.gerar_estatisticas_texto()

        print("\n" + "="*80)
        print("✅ RELATÓRIO COMPLETO GERADO!")
        print(f"📁 Arquivos em: {self.output_dir}")
        print("="*80)

    def gerar_estatisticas_texto(self):
        """Gera arquivo de texto com estatísticas descritivas"""
        print("\n📝 Gerando estatísticas textuais...")

        output_file = self.output_dir / "estatisticas.txt"

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ESTATÍSTICAS DESCRITIVAS - PIPELINE DE IMAGE STITCHING\n")
            f.write("="*80 + "\n\n")

            # Informações gerais
            f.write(f"Total de experimentos: {len(self.df)}\n")
            f.write(f"Experimentos com sucesso: {len(self.df[self.df['psnr_db'].notna()])}\n")
            f.write(f"Experimentos com erro: {len(self.df[self.df['psnr_db'].isna()])}\n\n")

            # Estatísticas por métrica
            df_success = self.df[self.df['psnr_db'].notna()]

            if len(df_success) > 0:
                f.write("PSNR (dB):\n")
                f.write(f"  Média: {df_success['psnr_db'].mean():.2f}\n")
                f.write(f"  Mediana: {df_success['psnr_db'].median():.2f}\n")
                f.write(f"  Desvio padrão: {df_success['psnr_db'].std():.2f}\n")
                f.write(f"  Mínimo: {df_success['psnr_db'].min():.2f}\n")
                f.write(f"  Máximo: {df_success['psnr_db'].max():.2f}\n\n")

                f.write("SSIM:\n")
                f.write(f"  Média: {df_success['ssim'].mean():.4f}\n")
                f.write(f"  Mediana: {df_success['ssim'].median():.4f}\n")
                f.write(f"  Desvio padrão: {df_success['ssim'].std():.4f}\n")
                f.write(f"  Mínimo: {df_success['ssim'].min():.4f}\n")
                f.write(f"  Máximo: {df_success['ssim'].max():.4f}\n\n")

                f.write("Tempo (segundos):\n")
                f.write(f"  Média: {df_success['tempo_segundos'].mean():.2f}\n")
                f.write(f"  Mediana: {df_success['tempo_segundos'].median():.2f}\n")
                f.write(f"  Desvio padrão: {df_success['tempo_segundos'].std():.2f}\n")
                f.write(f"  Mínimo: {df_success['tempo_segundos'].min():.2f}\n")
                f.write(f"  Máximo: {df_success['tempo_segundos'].max():.2f}\n\n")

                # Rankings
                f.write("="*80 + "\n")
                f.write("RANKINGS\n")
                f.write("="*80 + "\n\n")

                f.write("Top 3 por PSNR:\n")
                top_psnr = df_success.nlargest(3, 'psnr_db')[['conjunto', 'algoritmo', 'psnr_db']]
                for i, (_, row) in enumerate(top_psnr.iterrows(), 1):
                    f.write(f"  {i}. {row['algoritmo']} em {row['conjunto']}: {row['psnr_db']:.2f} dB\n")

                f.write("\nTop 3 por SSIM:\n")
                top_ssim = df_success.nlargest(3, 'ssim')[['conjunto', 'algoritmo', 'ssim']]
                for i, (_, row) in enumerate(top_ssim.iterrows(), 1):
                    f.write(f"  {i}. {row['algoritmo']} em {row['conjunto']}: {row['ssim']:.4f}\n")

                f.write("\nTop 3 mais rápidos:\n")
                top_rapidos = df_success.nsmallest(3, 'tempo_segundos')[['conjunto', 'algoritmo', 'tempo_segundos']]
                for i, (_, row) in enumerate(top_rapidos.iterrows(), 1):
                    f.write(f"  {i}. {row['algoritmo']} em {row['conjunto']}: {row['tempo_segundos']:.2f}s\n")

        print(f"✅ Estatísticas salvas: {output_file}")


# ===== CLI =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualizador de resultados do pipeline de image stitching"
    )
    parser.add_argument(
        "csv",
        help="Caminho para o CSV com resultados (ex: resultados_pipeline/resultados_metricas_*.csv)"
    )
    parser.add_argument(
        "--output",
        help="Diretório de saída (padrão: subpasta 'analises' no diretório do CSV)"
    )

    args = parser.parse_args()

    # Criar visualizador
    viz = VisualizadorResultados(args.csv)

    # Gerar relatório completo
    viz.gerar_relatorio_completo()

    print("\n✅ Análise completa!")
    print(f"\n📂 Arquivos gerados:")
    print(f"   - tabela_resumo.tex")
    print(f"   - tabela_detalhada.tex")
    if HAS_MATPLOTLIB:
        print(f"   - grafico_psnr.png")
        print(f"   - grafico_ssim.png")
        print(f"   - grafico_tempo.png")
        print(f"   - grafico_scatter_trade_off.png")
        print(f"   - heatmap_psnr.png")
    print(f"   - estatisticas.txt")
