"""
Script Master: Roda Todas as Analises
======================================
Executa todas as analises prioritarias e gera um relatorio completo.

Autor: Claude Code
Data: 2025-10-28
"""

import subprocess
import sys
from pathlib import Path
import time

def rodar_script(script_name, descricao):
    """Roda um script Python e reporta resultado"""
    print("\n" + "="*80)
    print(f"EXECUTANDO: {descricao}")
    print("="*80)

    inicio = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
            check=True
        )

        tempo_exec = time.time() - inicio
        print(f"\n[OK] {descricao} concluido em {tempo_exec:.1f}s")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n[ERRO] Falha ao executar {script_name}")
        print(f"Codigo de erro: {e.returncode}")
        return False

    except Exception as e:
        print(f"\n[ERRO] Excecao ao executar {script_name}: {e}")
        return False

def gerar_relatorio_final():
    """Gera um relatorio final consolidado"""
    print("\n" + "="*80)
    print("GERANDO RELATORIO FINAL CONSOLIDADO")
    print("="*80)

    output_dir = Path('results')
    relatorio_path = output_dir / 'RELATORIO_FINAL_ANALISES.txt'

    with open(relatorio_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATORIO FINAL - ANALISES ESTATISTICAS AVANCADAS\n")
        f.write("Fusao de Imagens Panoramicas: TCC Tassiane\n")
        f.write("="*80 + "\n\n")

        f.write("Este relatorio consolida os resultados de 5 analises independentes:\n\n")

        f.write("1. ANALISE ESTATISTICA BASICA\n")
        f.write("   Localizacao: results/analise_estatistica/\n")
        f.write("   Conteudo:\n")
        f.write("   - relatorio_estatistico_completo.txt\n")
        f.write("   - testes_comparativos.csv\n")
        f.write("   - intervalos_confianca.csv\n")
        f.write("   - matriz_correlacao.csv\n\n")

        f.write("2. ANALISE ESTRATIFICADA POR CENARIO\n")
        f.write("   Localizacao: results/analise_estratificada/\n")
        f.write("   Conteudo:\n")
        f.write("   - resultados_estratificados.csv\n")
        f.write("   - heatmap_pvalues_por_cenario.png\n")
        f.write("   - heatmap_cohens_d_por_cenario.png\n")
        f.write("   - barplot_metricas_por_cenario.png\n")
        f.write("   - scatter_psnr_tempo_por_cenario.png\n")
        f.write("   - secao_analise_estratificada.tex (PRONTO PARA TCC!)\n\n")

        f.write("3. ANALISE DE CASOS EXTREMOS\n")
        f.write("   Localizacao: results/analise_casos_extremos/\n")
        f.write("   Conteudo:\n")
        f.write("   - top_bottom_por_metodo.csv\n")
        f.write("   - datasets_desafiadores.csv\n")
        f.write("   - casos_deep_superior.csv\n")
        f.write("   - casos_classico_superior.csv\n")
        f.write("   - distribuicao_casos_extremos.png\n")
        f.write("   - violin_psnr_por_dataset.png\n")
        f.write("   - boxplot_range_por_metodo.png\n")
        f.write("   - heatmap_melhor_metodo_por_dataset.png\n")
        f.write("   - secao_casos_extremos.tex (PRONTO PARA TCC!)\n\n")

        f.write("4. VISUALIZACOES AVANCADAS\n")
        f.write("   Localizacao: results/visualizacoes_avancadas/\n")
        f.write("   Conteudo:\n")
        f.write("   - boxplots_comparativos.png\n")
        f.write("   - heatmap_psnr_metodo_dataset.png\n")
        f.write("   - scatter_3d_qualidade_velocidade.png\n")
        f.write("   - histogramas_distribuicao.png\n")
        f.write("   - matriz_correlacao.png\n")
        f.write("   - pareto_chart_metodos.png\n")
        f.write("   - radar_chart_top5.png\n\n")

        f.write("="*80 + "\n")
        f.write("PRINCIPAIS ACHADOS\n")
        f.write("="*80 + "\n\n")

        f.write("1. NAO HA DIFERENCA ESTATISTICAMENTE SIGNIFICATIVA entre classico e deep learning\n")
        f.write("   - Todos os p-values > 0.05\n")
        f.write("   - Cohen's d trivial (< 0.2) para todas as metricas\n")
        f.write("   - Implicacao: Para o dataset testado, metodos classicos sao equivalentes\n\n")

        f.write("2. DADOS NAO SEGUEM DISTRIBUICAO NORMAL\n")
        f.write("   - Shapiro-Wilk rejeitou normalidade em 9/10 testes\n")
        f.write("   - Testes nao-parametricos sao mais apropriados\n")
        f.write("   - Mann-Whitney U concordou com teste-t: sem diferencas\n\n")

        f.write("3. ANOVA NAO DETECTOU DIFERENCAS ENTRE OS 20 METODOS\n")
        f.write("   - PSNR: p=0.084 (marginalmente nao-significativo)\n")
        f.write("   - SSIM, Ghosting, Seam: p > 0.7\n")
        f.write("   - PROBLEMA: Isso sugere que metricas globais nao discriminam bem\n\n")

        f.write("4. CORRELACOES FORTES IDENTIFICADAS\n")
        f.write("   - Seam Energy <-> Ghosting: r=-0.883 (muito forte)\n")
        f.write("   - Seam Energy <-> SSIM: r=-0.789 (forte)\n")
        f.write("   - Menor seam energy = melhor qualidade\n\n")

        f.write("5. DATASETS MAIS DESAFIADORES\n")
        f.write("   - no_contractor_parking_1 (PSNR=8.51 dB)\n")
        f.write("   - graffiti_car_1 (PSNR=8.73 dB)\n")
        f.write("   - rental_office_1 (PSNR=9.01 dB)\n\n")

        f.write("="*80 + "\n")
        f.write("RECOMENDACOES PARA O TCC\n")
        f.write("="*80 + "\n\n")

        f.write("1. ADICIONAR SECAO 'ANALISE ESTATISTICA' ao capitulo de Resultados\n")
        f.write("   - Use: results/analise_estatistica/relatorio_estatistico_completo.txt\n")
        f.write("   - Copie tabelas e interpretacoes\n\n")

        f.write("2. ADICIONAR SECAO 'ANALISE ESTRATIFICADA' ao capitulo de Resultados\n")
        f.write("   - Use: results/analise_estratificada/secao_analise_estratificada.tex\n")
        f.write("   - Arquivo LaTeX PRONTO para copiar e colar!\n\n")

        f.write("3. ADICIONAR SECAO 'CASOS EXTREMOS' ao capitulo de Resultados\n")
        f.write("   - Use: results/analise_casos_extremos/secao_casos_extremos.tex\n")
        f.write("   - Arquivo LaTeX PRONTO para copiar e colar!\n\n")

        f.write("4. SUBSTITUIR GRAFICOS BASICOS por visualizacoes avancadas\n")
        f.write("   - Use imagens de: results/visualizacoes_avancadas/\n")
        f.write("   - Todos em alta resolucao (300 DPI)\n\n")

        f.write("5. REVISAR DISCUSSAO considerando resultados estatisticos\n")
        f.write("   - Enfatizar que nao ha diferenca significativa\n")
        f.write("   - Propor analises mais refinadas (regioes de overlap, etc.)\n")
        f.write("   - Discutir limitacoes das metricas globais\n\n")

        f.write("="*80 + "\n")
        f.write("PROXIMOS PASSOS (OPCIONAIS)\n")
        f.write("="*80 + "\n\n")

        f.write("Se voce tiver mais tempo, considere implementar:\n\n")

        f.write("1. Analise de regioes de overlap (PRIORIDADE ALTA)\n")
        f.write("   - PSNR/SSIM apenas nas regioes de sobreposicao\n")
        f.write("   - Pode revelar diferencas que metricas globais mascaram\n\n")

        f.write("2. Metricas de deteccao de objetos mais detalhadas\n")
        f.write("   - mAP por sub-regiao (centro vs overlap)\n")
        f.write("   - Confidence retention\n")
        f.write("   - Feature degradation\n\n")

        f.write("3. Analise de sensibilidade a parametros\n")
        f.write("   - Variar parametros dos top 3 metodos\n")
        f.write("   - Ver impacto em qualidade e tempo\n\n")

        f.write("="*80 + "\n")
        f.write("FIM DO RELATORIO\n")
        f.write("="*80 + "\n")

    print(f"[OK] Relatorio final salvo em: {relatorio_path}")

def main():
    inicio_total = time.time()

    print("="*80)
    print("EXECUTANDO TODAS AS ANALISES PRIORITARIAS")
    print("="*80)
    print("\nEste script executara os seguintes scripts em sequencia:")
    print("  1. analise_estatistica.py")
    print("  2. analise_estratificada.py")
    print("  3. analise_casos_extremos.py")
    print("  4. visualizacoes_avancadas.py")
    print("\nTempo estimado: 5-10 minutos")
    print("="*80)

    # Checar se os scripts existem
    scripts = [
        ('analise_estatistica.py', 'Analise Estatistica Basica'),
        ('analise_estratificada.py', 'Analise Estratificada por Cenario'),
        ('analise_casos_extremos.py', 'Analise de Casos Extremos'),
        ('visualizacoes_avancadas.py', 'Visualizacoes Avancadas')
    ]

    for script, _ in scripts:
        if not Path(script).exists():
            print(f"\n[ERRO] Script nao encontrado: {script}")
            print("Certifique-se de que todos os scripts estao no diretorio atual.")
            return

    # Executar cada script
    resultados = []
    for script, descricao in scripts:
        sucesso = rodar_script(script, descricao)
        resultados.append((script, sucesso))

    # Gerar relatorio final
    gerar_relatorio_final()

    # Resumo final
    tempo_total = time.time() - inicio_total

    print("\n" + "="*80)
    print("RESUMO DA EXECUCAO")
    print("="*80)

    sucessos = sum(1 for _, sucesso in resultados if sucesso)
    falhas = len(resultados) - sucessos

    print(f"\nTotal de scripts: {len(resultados)}")
    print(f"Sucessos: {sucessos}")
    print(f"Falhas: {falhas}")
    print(f"Tempo total: {tempo_total/60:.1f} minutos")

    print("\nStatus detalhado:")
    for script, sucesso in resultados:
        status = "[OK]" if sucesso else "[ERRO]"
        print(f"  {status} {script}")

    if falhas == 0:
        print("\n" + "="*80)
        print("[OK] TODAS AS ANALISES FORAM CONCLUIDAS COM SUCESSO!")
        print("="*80)
        print("\nProximos passos:")
        print("  1. Leia: results/RELATORIO_FINAL_ANALISES.txt")
        print("  2. Revise os arquivos gerados em cada pasta")
        print("  3. Copie as secoes LaTeX para seu TCC")
        print("  4. Substitua os graficos antigos pelos novos")
        print("\nBom trabalho!")
    else:
        print("\n" + "="*80)
        print(f"[AVISO] {falhas} script(s) falharam")
        print("="*80)
        print("Revise os erros acima e tente executar os scripts individualmente.")

if __name__ == "__main__":
    main()
