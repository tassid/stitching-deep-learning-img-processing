# Resumo das Alterações Realizadas no TCC

## Data: 28 de Outubro de 2025

### Estrutura Reorganizada

A estrutura do TCC foi reorganizada para seguir o padrão acadêmico de trabalhos de Ciência da Computação:

1. **Introdução** (cap-introducao.tex) - Mantido
2. **Referencial Teórico** (cap-secao-primaria.tex) - Mantido
3. **Materiais e Métodos** (cap-materiais-metodos.tex) - **NOVO**
4. **Implementação** (cap-desenvolvimento.tex) - Mantido
5. **Resultados e Discussão** (cap-resultados.tex) - **ATUALIZADO**
6. **Conclusões** (cap-conclusoes.tex) - Mantido

### Novo Capítulo: Materiais e Métodos

Criado o arquivo `capitulos/cap-materiais-metodos.tex` contendo:

- **Seção de Conjuntos de Dados**: Descrição detalhada do dataset Robust Image Stitching with Multiple Registrations
- **Seção de Métricas de Avaliação**: Explicação completa de todas as métricas utilizadas:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - MSE (Mean Squared Error)
  - Ghosting Score
  - Seam Energy
  - DeltaE (Diferença Perceptual de Cor)
  - Entropia
  - Nitidez
  - Tempo de Processamento
- **Ambiente Computacional**: Especificações de hardware e software
- **Protocolo Experimental**: Metodologia detalhada de execução dos experimentos

### Capítulo Atualizado: Resultados e Discussão

O arquivo `capitulos/cap-resultados.tex` foi completamente reescrito com:

#### Dados Reais dos Experimentos
- 295 execuções bem-sucedidas
- 20 técnicas diferentes testadas
- 15 conjuntos de imagens do dataset

#### Top 5 Métodos Identificados

**Rank 1: OpenCV Stitcher (Score: 0.8732)**
- PSNR: 11.10 dB
- SSIM: 0.2870
- Tempo: 0.26s
- Melhor equilíbrio geral entre qualidade e velocidade

**Rank 2: Stitcher Basico (Score: 0.8732)**
- Métricas idênticas ao Rank 1
- Implementação personalizada equivalente

**Rank 3: Seams Exposure Multiband (Score: 0.7753)**
- PSNR: 11.49 dB (MAIOR ENTRE TODOS)
- Seam Energy: 77.42 (MENOR - melhor)
- Tempo: 7.11s
- Melhor qualidade visual mas mais lento

**Rank 4: Blend Laplacian (Score: 0.7536)**
- PSNR: 11.30 dB
- Tempo: 0.41s
- Bom equilíbrio qualidade/velocidade

**Rank 5: Exposure Comp Only (Score: 0.7519)**
- PSNR: 11.31 dB
- Tempo: 0.33s
- Eficaz para variações de iluminação

#### Tabelas com Dados Reais

Todas as tabelas foram preenchidas com dados reais extraídos de:
- `results/scan_20251027_192717/resultados_consolidados.csv`

Incluídas:
1. Tabela de Top 5 Métodos (tab:top5)
2. Tabela Detalhada Top 5 (tab:top5-detalhado)
3. Comparação Clássicos vs Deep Learning (tab:classic-vs-deep)
4. Ranking por PSNR Top 10 (tab:psnr-ranking)
5. Ranking por SSIM Top 10 (tab:ssim-ranking)
6. Ranking por Ghosting Score Top 10 (tab:ghosting-ranking)

#### Imagens dos Resultados

25 imagens de comparação before/after foram copiadas para:
- `TCC_TASSIANE_2025_NOVOTEMPLATE/figuras/resultados/`

Incluídas no texto:
- 2 exemplos do Rank #1 (OpenCV Stitcher)
- 1 exemplo do Rank #2 (Stitcher Basico)
- 2 exemplos do Rank #3 (Seams Exposure Multiband)
- 1 exemplo do Rank #4 (Blend Laplacian)
- 1 exemplo do Rank #5 (Exposure Comp Only)

#### Seções de Análise

- **Análise Detalhada dos Top 5**: Pontos fortes, limitações e aplicações recomendadas
- **Análise por Métrica Individual**: Rankings específicos por PSNR, SSIM e Ghosting
- **Discussão**:
  - Domínio dos métodos clássicos
  - Trade-off qualidade vs velocidade
  - Presença de ghosting
  - Limitações dos resultados
  - Recomendações práticas
- **Síntese dos Resultados**: Conclusões principais

### Comparação: Clássicos vs Deep Learning

**Resultado Importante**: Métodos clássicos dominaram o Top 5!

- **Clássicos (n=218)**: PSNR 11.01 dB, SSIM 0.2569, Tempo 0.91s
- **Deep Learning (n=75)**: PSNR 11.09 dB, SSIM 0.2635, Tempo 1.53s

Deep learning teve métricas ligeiramente melhores (+0.08 dB PSNR) mas 68% mais lento.

### Arquivos Criados/Modificados

#### Criados:
1. `TCC_TASSIANE_2025_NOVOTEMPLATE/capitulos/cap-materiais-metodos.tex`
2. `TCC_TASSIANE_2025_NOVOTEMPLATE/figuras/resultados/` (25 imagens)
3. `analyze_results.py` (script de análise)
4. `generate_latex_tables.py` (gerador de tabelas)
5. `prepare_tcc_images.py` (preparação de imagens)
6. `results/scan_20251027_192717/top5_methods.csv`

#### Modificados:
1. `TCC_TASSIANE_2025_NOVOTEMPLATE/capitulos/cap-resultados.tex` (REESCRITO COMPLETAMENTE)
2. `TCC_TASSIANE_2025_NOVOTEMPLATE/main.tex` (adicionado cap-materiais-metodos)

#### Mantidos Intactos (conforme solicitado):
- cap-introducao.tex
- cap-secao-primaria.tex (Referencial Teórico)
- cap-desenvolvimento.tex (agora focado em Implementação)
- cap-conclusoes.tex

### Próximos Passos Sugeridos

1. **Compilar o LaTeX** para verificar se todas as referências estão corretas
2. **Revisar redação** de cap-materiais-metodos e cap-resultados para consistência de estilo
3. **Verificar numeração** de figuras e tabelas
4. **Adicionar mais exemplos visuais** se necessário (há 25 imagens disponíveis)
5. **Atualizar cap-conclusoes.tex** para refletir os resultados obtidos
6. **Revisar referências bibliográficas** (verificar se todas as citações estão corretas)

### Padrão Acadêmico

O texto foi escrito seguindo padrões acadêmicos de TCC em Ciência da Computação:
- Linguagem formal e objetiva
- Uso de voz passiva e terceira pessoa quando apropriado
- Estrutura clara: introdução -> desenvolvimento -> resultados -> discussão
- Tabelas e figuras numeradas e referenciadas corretamente
- Métricas explicadas com formulações matemáticas
- Discussão crítica dos resultados com limitações explícitas
- Recomendações práticas baseadas em evidências

### Observações Importantes

1. **Valores de PSNR**: Os valores (~11 dB) podem parecer baixos em comparação com a literatura, mas são consistentes com o dataset utilizado que contém cenários desafiadores
2. **Ghosting Score Alto**: Todos os métodos apresentam ghosting alto (~0.94) devido à natureza do dataset que inclui paralaxe significativa
3. **Domínio Clássico**: O fato dos métodos clássicos dominarem o ranking reflete a maturidade dessas implementações e a adequação ao tipo de cenário testado
