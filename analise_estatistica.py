"""
Analise Estatística Avancada dos Resultados de Fusao de Imagens
Autor: Claude Code
Data: 2025-10-28
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, mannwhitneyu, shapiro, levene
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class AnalisadorEstatistico:
    def __init__(self, csv_path):
        """Inicializa o analisador com os dados"""
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        # Remover linhas com erro
        self.df = self.df[self.df['status'] == 'sucesso'].copy()

        # Adicionar categoria (classico vs deep)
        self.df['categoria'] = self.df['metodo'].apply(
            lambda x: 'Classico' if 'classic' in x else 'Deep Learning'
        )

        print(f"[OK] Dados carregados: {len(self.df)} execucoes bem-sucedidas")
        print(f"  - Metodos classicos: {sum(self.df['categoria'] == 'Classico')}")
        print(f"  - Metodos deep learning: {sum(self.df['categoria'] == 'Deep Learning')}")
        print(f"  - Conjuntos de imagens: {self.df['pasta'].nunique()}")

    def analise_descritiva_completa(self):
        """Estatisticas descritivas detalhadas"""
        print("\n" + "="*80)
        print("1. ESTATISTICAS DESCRITIVAS COMPLETAS")
        print("="*80)

        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos', 'entropia']

        # Por categoria
        print("\n### Por Categoria (Classico vs Deep Learning)")
        stats_categoria = self.df.groupby('categoria')[metricas].agg([
            'count', 'mean', 'std', 'min', 'max',
            ('q25', lambda x: np.percentile(x, 25)),
            ('median', 'median'),
            ('q75', lambda x: np.percentile(x, 75)),
            ('cv', lambda x: (np.std(x) / np.mean(x) * 100) if np.mean(x) != 0 else 0)  # Coeficiente de variacao
        ])

        print(stats_categoria.to_string())

        # Por metodo (top 10)
        print("\n### Top 10 Metodos (por PSNR mehdio)")
        top10 = self.df.groupby('label')[metricas].agg(['mean', 'std']).sort_values(
            ('psnr_db', 'mean'), ascending=False
        ).head(10)
        print(top10.to_string())

        return stats_categoria

    def teste_normalidade(self):
        """Testa normalidade das distribuições (Shapiro-Wilk)"""
        print("\n" + "="*80)
        print("2. TESTES DE NORMALIDADE (Shapiro-Wilk)")
        print("="*80)
        print("H0: Os dados seguem distribuição normal")
        print("Se p-value < 0.05, rejeita-se H0 (dados NÃO são normais)\n")

        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos']
        resultados = []

        for categoria in ['Classico', 'Deep Learning']:
            dados_cat = self.df[self.df['categoria'] == categoria]
            print(f"\n### {categoria}")

            for metrica in metricas:
                stat, p_value = shapiro(dados_cat[metrica].dropna())
                normal = "[OK] Normal" if p_value > 0.05 else "[X] NÃO Normal"
                resultados.append({
                    'Categoria': categoria,
                    'Metrica': metrica,
                    'Estatística': stat,
                    'p-value': p_value,
                    'Normalidade': normal
                })
                print(f"  {metrica:20s}: W={stat:.4f}, p={p_value:.4e} -> {normal}")

        return pd.DataFrame(resultados)

    def teste_homogeneidade_variancias(self):
        """Teste de Levene para homogeneidade de variancias"""
        print("\n" + "="*80)
        print("3. TESTE DE HOMOGENEIDADE DE VARIÂNCIAS (Levene)")
        print("="*80)
        print("H0: As variancias são iguais entre grupos")
        print("Se p-value < 0.05, rejeita-se H0 (variancias são DIFERENTES)\n")

        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos']
        resultados = []

        classico = self.df[self.df['categoria'] == 'Classico']
        deep = self.df[self.df['categoria'] == 'Deep Learning']

        for metrica in metricas:
            stat, p_value = levene(classico[metrica].dropna(), deep[metrica].dropna())
            homogeneo = "[OK] Homogeneo" if p_value > 0.05 else "[X] Heterogeneo"
            resultados.append({
                'Metrica': metrica,
                'Estatística': stat,
                'p-value': p_value,
                'Resultado': homogeneo
            })
            print(f"  {metrica:20s}: W={stat:.4f}, p={p_value:.4e} -> {homogeneo}")

        return pd.DataFrame(resultados)

    def testes_comparativos(self):
        """Testes t e Mann-Whitney para comparar classico vs deep learning"""
        print("\n" + "="*80)
        print("4. TESTES COMPARATIVOS: Classico vs Deep Learning")
        print("="*80)

        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos']
        resultados = []

        classico = self.df[self.df['categoria'] == 'Classico']
        deep = self.df[self.df['categoria'] == 'Deep Learning']

        for metrica in metricas:
            # Teste t (paramehtrico)
            t_stat, t_pvalue = ttest_ind(
                classico[metrica].dropna(),
                deep[metrica].dropna(),
                equal_var=False  # Welch's t-test (não assume variancias iguais)
            )

            # Mann-Whitney U (não-paramehtrico)
            u_stat, u_pvalue = mannwhitneyu(
                classico[metrica].dropna(),
                deep[metrica].dropna(),
                alternative='two-sided'
            )

            # Calcular tamanho do efeito (Cohen's d)
            mean_diff = classico[metrica].mean() - deep[metrica].mean()
            pooled_std = np.sqrt(
                (classico[metrica].std()**2 + deep[metrica].std()**2) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0

            # Classificação do tamanho do efeito
            if abs(cohens_d) < 0.2:
                effect_size = "Trivial"
            elif abs(cohens_d) < 0.5:
                effect_size = "Pequeno"
            elif abs(cohens_d) < 0.8:
                effect_size = "Mehdio"
            else:
                effect_size = "Grande"

            significativo_t = "[OK] SIM" if t_pvalue < 0.05 else "[X] NÃO"
            significativo_u = "[OK] SIM" if u_pvalue < 0.05 else "[X] NÃO"

            resultados.append({
                'Metrica': metrica,
                'Media_Classico': classico[metrica].mean(),
                'Media_Deep': deep[metrica].mean(),
                'Diferença': mean_diff,
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'Sig_t': significativo_t,
                'U_statistic': u_stat,
                'U_pvalue': u_pvalue,
                'Sig_U': significativo_u,
                'Cohens_d': cohens_d,
                'Tamanho_Efeito': effect_size
            })

            print(f"\n### {metrica}")
            print(f"  Media Classico: {classico[metrica].mean():.4f}")
            print(f"  Media Deep:     {deep[metrica].mean():.4f}")
            print(f"  Diferença:      {mean_diff:+.4f}")
            print(f"  Teste t: t={t_stat:.3f}, p={t_pvalue:.4e} -> {significativo_t}")
            print(f"  Mann-Whitney: U={u_stat:.0f}, p={u_pvalue:.4e} -> {significativo_u}")
            print(f"  Cohen's d: {cohens_d:.3f} ({effect_size})")

        return pd.DataFrame(resultados)

    def anova_por_metodo(self):
        """ANOVA para comparar múltiplos metodos"""
        print("\n" + "="*80)
        print("5. ANOVA: Comparação entre Todos os Metodos")
        print("="*80)
        print("H0: Todas as medias são iguais")
        print("Se p-value < 0.05, rejeita-se H0 (pelo menos uma media eh diferente)\n")

        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy']
        resultados = []

        # Agrupar por metodo
        grupos = [grupo[1] for grupo in self.df.groupby('metodo')]

        for metrica in metricas:
            # Preparar dados por grupo
            dados_grupos = [grupo[metrica].dropna() for grupo in grupos if len(grupo[metrica].dropna()) > 0]

            # ANOVA
            f_stat, p_value = f_oneway(*dados_grupos)

            # Calcular eta-squared (tamanho do efeito)
            grand_mean = self.df[metrica].mean()
            ss_between = sum([len(g) * (g.mean() - grand_mean)**2 for g in dados_grupos])
            ss_total = sum([(x - grand_mean)**2 for g in dados_grupos for x in g])
            eta_squared = ss_between / ss_total if ss_total != 0 else 0

            significativo = "[OK] SIM" if p_value < 0.05 else "[X] NÃO"

            resultados.append({
                'Metrica': metrica,
                'F_statistic': f_stat,
                'p_value': p_value,
                'Significativo': significativo,
                'eta_squared': eta_squared
            })

            print(f"  {metrica:20s}: F={f_stat:.3f}, p={p_value:.4e}, eta²={eta_squared:.4f} -> {significativo}")

        return pd.DataFrame(resultados)

    def intervalos_confianca(self, confianca=0.95):
        """Calcula intervalos de confianca para cada metrica"""
        print("\n" + "="*80)
        print(f"6. INTERVALOS DE CONFIANÇA ({int(confianca*100)}%)")
        print("="*80)

        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos']
        resultados = []

        alpha = 1 - confianca

        for categoria in ['Classico', 'Deep Learning']:
            dados_cat = self.df[self.df['categoria'] == categoria]
            print(f"\n### {categoria}")

            for metrica in metricas:
                valores = dados_cat[metrica].dropna()
                n = len(valores)
                media = valores.mean()
                std_err = valores.std() / np.sqrt(n)

                # Intervalo de confianca usando distribuição t
                df = n - 1
                t_critical = stats.t.ppf(1 - alpha/2, df)
                margem_erro = t_critical * std_err

                ic_inferior = media - margem_erro
                ic_superior = media + margem_erro

                resultados.append({
                    'Categoria': categoria,
                    'Metrica': metrica,
                    'n': n,
                    'Media': media,
                    'Std_Error': std_err,
                    'IC_Inferior': ic_inferior,
                    'IC_Superior': ic_superior,
                    'Margem_Erro': margem_erro
                })

                print(f"  {metrica:20s}: {media:.4f} ± {margem_erro:.4f} [{ic_inferior:.4f}, {ic_superior:.4f}]")

        return pd.DataFrame(resultados)

    def analise_correlacao(self):
        """Analise de correlacao entre metricas"""
        print("\n" + "="*80)
        print("7. MATRIZ DE CORRELAÇÃO (Pearson)")
        print("="*80)

        metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos', 'entropia']

        # Correlacao de Pearson
        corr_matrix = self.df[metricas].corr(method='pearson')
        print("\n### Correlacoes de Pearson:")
        print(corr_matrix.to_string())

        # Identificar correlacoes fortes
        print("\n### Correlacoes Fortes (|r| > 0.7):")
        for i in range(len(metricas)):
            for j in range(i+1, len(metricas)):
                r = corr_matrix.iloc[i, j]
                if abs(r) > 0.7:
                    print(f"  {metricas[i]:20s} <-> {metricas[j]:20s}: r={r:+.3f}")

        return corr_matrix

    def analise_por_dataset(self):
        """Analise de variabilidade por conjunto de imagens"""
        print("\n" + "="*80)
        print("8. ANALISE POR CONJUNTO DE IMAGENS")
        print("="*80)

        # Calcular coeficiente de variacao por dataset
        datasets_stats = self.df.groupby('pasta').agg({
            'psnr_db': ['mean', 'std', 'count'],
            'ssim': ['mean', 'std'],
            'tempo_segundos': ['mean', 'std']
        })

        # Calcular CV
        datasets_stats[('psnr_db', 'cv')] = (
            datasets_stats[('psnr_db', 'std')] / datasets_stats[('psnr_db', 'mean')] * 100
        )

        print("\n### Variabilidade por Dataset (ordenado por CV do PSNR):")
        datasets_sorted = datasets_stats.sort_values(('psnr_db', 'cv'), ascending=False)
        print(datasets_sorted.to_string())

        # Identificar datasets mais desafiadores
        print("\n### Datasets Mais Desafiadores (menor PSNR mehdio):")
        desafiadores = datasets_stats.sort_values(('psnr_db', 'mean')).head(5)
        print(desafiadores.to_string())

        return datasets_stats

    def gerar_relatorio_completo(self, output_dir='results/analise_estatistica'):
        """Gera relatorio completo em formato texto"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Criar arquivo de relatorio
        relatorio_path = output_path / 'relatorio_estatistico_completo.txt'

        with open(relatorio_path, 'w', encoding='utf-8') as f:
            # Redirecionar print para arquivo
            import sys
            old_stdout = sys.stdout
            sys.stdout = f

            print("="*80)
            print("RELATORIO DE ANALISE ESTATISTICA COMPLETA")
            print("Fusao de Imagens Panorâmicas: Metodos Classicos vs Deep Learning")
            print("="*80)
            print(f"Data: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total de execucoes: {len(self.df)}")
            print("="*80)

            # Executar todas as analises
            self.analise_descritiva_completa()
            self.teste_normalidade()
            self.teste_homogeneidade_variancias()
            self.testes_comparativos()
            self.anova_por_metodo()
            self.intervalos_confianca()
            self.analise_correlacao()
            self.analise_por_dataset()

            # Restaurar stdout
            sys.stdout = old_stdout

        print(f"\n[OK] Relatorio salvo em: {relatorio_path}")

        # Salvar DataFrames em CSV
        df_testes = self.testes_comparativos()
        df_testes.to_csv(output_path / 'testes_comparativos.csv', index=False)

        df_ic = self.intervalos_confianca()
        df_ic.to_csv(output_path / 'intervalos_confianca.csv', index=False)

        corr_matrix = self.analise_correlacao()
        corr_matrix.to_csv(output_path / 'matriz_correlacao.csv')

        print(f"[OK] CSVs salvos em: {output_path}")

def main():
    # Carregar dados
    csv_path = 'results/scan_20251027_192717/resultados_consolidados.csv'

    print("="*80)
    print("ANALISE ESTATISTICA AVANÇADA - FUSÃO DE IMAGENS")
    print("="*80)

    analisador = AnalisadorEstatistico(csv_path)

    # Executar todas as analises e exibir no console
    analisador.analise_descritiva_completa()
    analisador.teste_normalidade()
    analisador.teste_homogeneidade_variancias()
    analisador.testes_comparativos()
    analisador.anova_por_metodo()
    analisador.intervalos_confianca()
    analisador.analise_correlacao()
    analisador.analise_por_dataset()

    # Gerar relatorio completo
    print("\n" + "="*80)
    print("GERANDO RELATORIO COMPLETO...")
    print("="*80)
    analisador.gerar_relatorio_completo()

    print("\n" + "="*80)
    print("[OK] ANALISE CONCLUÍDA COM SUCESSO!")
    print("="*80)

if __name__ == "__main__":
    main()
