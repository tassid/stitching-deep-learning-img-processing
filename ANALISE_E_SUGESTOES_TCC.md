# Análise e Sugestões de Melhoria - TCC

Data: 28 de Outubro de 2025

## RESUMO DA ANÁLISE

**Status Geral**: O TCC possui uma estrutura sólida com experimentos completos e resultados robustos. No entanto, há alguns problemas críticos que precisam ser resolvidos antes da defesa.

**Pontuação Geral**: 7.5/10

### Pontos Fortes ✓
- Experimentos bem executados (295 execuções, 20 técnicas)
- Capítulo de Resultados completo com dados reais e análises visuais
- Materiais e Métodos bem estruturado com métricas explicadas
- Referencial Teórico abrangente (42 citações)
- Todas as referências citadas presentes no BibTeX (66 citações verificadas)
- 25 imagens de resultados disponíveis

### Problemas Críticos ✗
1. **Capítulo de Conclusões VAZIO** (apenas título)
2. **Resumo e Abstract SÓ COM TEMPLATE**
3. **Desconexão entre objetivos e resultados**
4. **Introdução confunde fusão de imagens com detecção de objetos**

---

## SUGESTÕES POR PRIORIDADE

### 🔴 PRIORIDADE CRÍTICA (Fazer ANTES de tudo)

#### 1. ESCREVER O CAPÍTULO DE CONCLUSÕES ⚠️
**Arquivo**: `capitulos/cap-conclusoes.tex`
**Problema**: Capítulo está completamente vazio, apenas com o título
**Impacto**: Sem conclusões, o TCC está INCOMPLETO e não pode ser defendido

**O que deve conter**:
```
1. Retomada dos objetivos (2-3 parágrafos)
   - Objetivo geral: avaliar técnicas de fusão de imagens
   - Objetivo específico 1: implementar técnicas clássicas e deep learning
   - Objetivo específico 2: comparar desempenho dos métodos

2. Principais resultados obtidos (3-4 parágrafos)
   - OpenCV Stitcher e Stitcher Básico obtiveram melhor score geral (0.8732)
   - Seams Exposure Multiband teve maior PSNR (11.49 dB)
   - Métodos clássicos vs deep learning: diferença pequena (+0.08 dB PSNR)
   - Deep learning 68% mais lento que clássicos
   - Ghosting Score alto (0.94) em ambas abordagens indica desafio do dataset

3. Contribuições do trabalho (2 parágrafos)
   - Comparação sistemática de 20 técnicas diferentes
   - Identificação de trade-offs qualidade vs velocidade
   - Análise quantitativa e qualitativa com métricas múltiplas

4. Limitações do estudo (1-2 parágrafos)
   - Dataset específico (Multiple Registrations)
   - PSNR relativamente baixo (~11 dB)
   - Ghosting presente em todos os métodos
   - Não testado em cenários com detecção de objetos

5. Trabalhos futuros (1-2 parágrafos)
   - Testar em outros datasets (UDIS-D, NISwGSP)
   - Implementar técnicas híbridas (clássico + deep learning)
   - Avaliar em aplicações reais de detecção de objetos
   - Otimizar tempo de processamento dos métodos deep learning
   - Desenvolver métricas específicas para mosaicos com objetos
```

**Extensão recomendada**: 3-4 páginas

---

#### 2. ESCREVER RESUMO E ABSTRACT ⚠️
**Arquivos**:
- `PreTexto/resumo.tex`
- `PreTexto/abstract.tex`

**Problema**: Ambos contém apenas instruções do template
**Impacto**: Documentos obrigatórios estão vazios

**Template de Resumo (150-500 palavras)**:
```
Este trabalho apresenta uma análise comparativa de técnicas de fusão de imagens
para construção de mosaicos panorâmicos, avaliando métodos clássicos e baseados
em aprendizado profundo. [Contexto e motivação]. O objetivo foi implementar e
comparar 20 técnicas diferentes utilizando o dataset Robust Image Stitching with
Multiple Registrations. [Metodologia]. Foram realizadas 295 execuções bem-sucedidas,
com cálculo de métricas PSNR, SSIM, Ghosting Score, Seam Energy, e tempo de
processamento. [Resultados principais]. Os métodos OpenCV Stitcher e Stitcher
Básico obtiveram melhor desempenho geral (score 0.8732), com PSNR de 11.10 dB
e tempo de 0.26s. O método Seams Exposure Multiband apresentou o maior PSNR
(11.49 dB) mas com tempo significativamente maior (7.11s). A comparação entre
abordagens clássicas e deep learning mostrou diferença pequena em qualidade
(+0.08 dB PSNR para deep learning) mas com custo computacional 68% maior.
[Conclusão]. Os resultados demonstram que técnicas clássicas oferecem excelente
relação custo-benefício para fusão de imagens panorâmicas, enquanto métodos de
aprendizado profundo podem ser mais adequados quando qualidade é prioridade
absoluta e tempo não é limitante.
```

**Palavras-chave sugeridas**:
- fusão de imagens
- image stitching
- visão computacional
- aprendizado profundo
- panoramas

---

#### 3. ALINHAR OBJETIVOS COM RESULTADOS ⚠️
**Problema**: Introdução fala muito de "detecção de objetos", mas trabalho foca 100% em fusão/stitching

**Duas opções**:

**OPÇÃO A - RECOMENDADA**: Ajustar introdução e objetivos para focar APENAS em fusão de imagens
- Remover ou reduzir drasticamente menções a "detecção de objetos"
- Reescrever objetivo geral focando em fusão/stitching
- Manter coerência do início ao fim do documento

**OPÇÃO B**: Adicionar experimentos de detecção de objetos
- Aplicar YOLO nas imagens originais vs panoramas
- Comparar taxa de detecção
- Avaliar como fusão afeta detecção
- **Não recomendado**: Adicionaria muito trabalho e está fora do escopo atual

**Sugestão de novo objetivo geral** (OPÇÃO A):
```
O objetivo geral deste trabalho é investigar, implementar e comparar técnicas
de fusão de imagens para construção de mosaicos panorâmicos, avaliando métodos
clássicos e baseados em aprendizado profundo quanto à qualidade visual, presença
de artefatos e eficiência computacional. Este objetivo principal desdobra-se nos
seguintes objetivos específicos:

- Implementar e testar técnicas clássicas e baseadas em aprendizado profundo
  para fusão de imagens panorâmicas utilizando datasets públicos
- Avaliar quantitativa e qualitativamente o desempenho dos métodos através de
  métricas como PSNR, SSIM, Ghosting Score, Seam Energy e tempo de processamento
- Identificar trade-offs entre qualidade, presença de artefatos e custo
  computacional para orientar a escolha de técnicas em aplicações práticas
```

---

### 🟡 PRIORIDADE ALTA (Fazer na sequência)

#### 4. REVISAR INTRODUÇÃO
**Arquivo**: `capitulos/cap-introducao.tex`

**Problemas identificados**:
1. Primeiros 5 parágrafos focam muito em detecção de objetos
2. Texto comentado (linhas 21-29) está confuso - remover ou incorporar
3. Falta conectar a introdução com o que foi realmente feito

**Sugestões**:
- Reescrever parágrafo 1-3 focando em fusão de imagens como técnica principal
- Mencionar detecção de objetos apenas como UMA das aplicações (não a principal)
- Adicionar parágrafo sobre desafios de stitching (ghosting, paralaxe, alinhamento)
- Remover todo o código comentado (linhas 21-29 e 40-47)

**Nova estrutura sugerida**:
```
1. Contextualização: fusão de imagens e aplicações
2. Importância de mosaicos panorâmicos
3. Desafios técnicos (ghosting, alinhamento, qualidade)
4. Evolução: métodos clássicos → deep learning
5. Lacuna: falta comparação sistemática
6. Proposta deste trabalho
7. Seção de objetivos
8. Seção de justificativa
9. Seção de estrutura do trabalho
```

---

#### 5. MELHORAR CAPÍTULO DE RESULTADOS
**Arquivo**: `capitulos/cap-resultados.tex`

**Pontos positivos**: Já está muito bom, mas pode melhorar

**Sugestões**:
1. Adicionar mais exemplos visuais (usar as 25 imagens disponíveis)
   - Atualmente só 7 imagens são usadas
   - Adicionar comparação lado-a-lado: clássico vs deep learning
   - Mostrar casos de falha/ghosting evidente

2. Criar subseção "Análise de Casos Extremos"
   - Melhor resultado obtido
   - Pior resultado obtido
   - Caso com maior ghosting
   - Caso mais rápido vs mais lento

3. Adicionar gráficos comparativos
   - Scatter plot: PSNR vs Tempo
   - Bar chart: Comparação de todas as 20 técnicas
   - Box plot: Distribuição de SSIM por categoria

4. Expandir discussão sobre valores baixos de PSNR
   - Por que ~11 dB é baixo?
   - É específico do dataset?
   - Comparar com literatura

**Código LaTeX para figura comparativa**:
```latex
\begin{figure}[htb]
\centering
\begin{subfigure}{0.45\textwidth}
    \includegraphics[width=\textwidth]{figuras/resultados/metodo_classico_exemplo.jpg}
    \caption{Método Clássico (OpenCV Stitcher)}
\end{subfigure}
\hfill
\begin{subfigure}{0.45\textwidth}
    \includegraphics[width=\textwidth]{figuras/resultados/metodo_deep_exemplo.jpg}
    \caption{Método Deep Learning (LoFTR)}
\end{subfigure}
\caption{Comparação visual: métodos clássicos vs deep learning}
\label{fig:classic-vs-deep-visual}
\end{figure}
```

---

#### 6. VERIFICAR CONSISTÊNCIA ENTRE CAPÍTULOS
**Problema**: Há menções desencontradas entre capítulos

**Checklist de consistência**:
- [ ] Introdução menciona os mesmos métodos do Referencial Teórico?
- [ ] Materiais e Métodos descreve as técnicas implementadas?
- [ ] Resultados analisa TODAS as técnicas mencionadas antes?
- [ ] Conclusão retoma TODOS os objetivos da Introdução?
- [ ] Dataset descrito em Materiais coincide com o usado?

**Ação**: Fazer leitura completa anotando inconsistências

---

### 🟢 PRIORIDADE MÉDIA (Melhorias incrementais)

#### 7. ADICIONAR FIGURAS DO REFERENCIAL TEÓRICO
**Arquivo**: `capitulos/cap-secao-primaria.tex`

**Problema**: Capítulo longo e denso, poderia ter mais ilustrações

**Sugestões**:
- Diagrama de pipeline típico de image stitching
- Exemplos visuais de detecção de keypoints (SIFT, ORB)
- Ilustração de transformações (homografia)
- Diagrama de arquitetura de rede deep learning usada
- Exemplos de diferentes tipos de blending

**Ferramentas para criar diagramas**:
- draw.io
- TikZ (LaTeX)
- PowerPoint + export PNG

---

#### 8. MELHORAR TABELAS
**Sugestões**:
1. Adicionar tabela com TODAS as 20 técnicas (não só top 5)
2. Destacar células com melhor valor (negrito)
3. Adicionar coluna de "Categoria" (Clássico/Deep Learning)
4. Usar cores sutis para destacar top 3

**Exemplo LaTeX**:
```latex
\begin{tabular}{clcccc}
\rowcolor{lightgray}
\textbf{Rank} & \textbf{Método} & \textbf{Cat.} & \textbf{PSNR} & \textbf{SSIM} & \textbf{Tempo} \\
\rowcolor{green!10}
1 & OpenCV Stitcher & Clássico & \textbf{11.10} & \textbf{0.2870} & 0.26 \\
...
\end{tabular}
```

---

#### 9. REVISAR REDAÇÃO
**Problemas encontrados**:
- Alguns parágrafos muito longos (>10 linhas)
- Uso excessivo de voz passiva em alguns trechos
- Repetição de termos ("fusão de imagens" aparece muito)

**Dicas de redação acadêmica**:
1. Parágrafos: 4-8 linhas é ideal
2. Variar estruturas: ativo + passivo
3. Usar sinônimos: "image stitching", "construção de mosaicos", "fusão panorâmica"
4. Conectores entre parágrafos: "Além disso", "Contudo", "Neste contexto"

---

#### 10. ADICIONAR APÊNDICES/ANEXOS
**Sugestões de conteúdo**:

**Apêndice A - Parâmetros de Configuração**
- Todos os hiperparâmetros usados em cada técnica
- Configurações de hardware/software detalhadas

**Apêndice B - Tabela Completa de Resultados**
- Resultados de TODAS as 295 execuções
- Permite reprodutibilidade total

**Apêndice C - Código-fonte**
- Principais funções implementadas
- Scripts de análise de dados

**Anexo A - Documentação dos Datasets**
- Descrição completa do dataset usado
- Exemplos de imagens originais

---

### 🔵 PRIORIDADE BAIXA (Nice to have)

#### 11. Melhorias Cosméticas
- [ ] Revisar formatação de todas as equações
- [ ] Verificar espaçamento entre seções
- [ ] Padronizar estilo de citações (autor-data vs numérico)
- [ ] Adicionar lista de figuras e tabelas (já existe)
- [ ] Revisar numeração de páginas

#### 12. Glossário de Termos
- Criar glossário com termos técnicos
- SIFT, ORB, RANSAC, homografia, ghosting, etc.

#### 13. Revisão de Inglês (Abstract)
- Contratar revisor nativo ou usar ferramenta profissional
- Grammarly Premium
- Serviço de revisão da universidade

---

## CHECKLIST FINAL ANTES DA ENTREGA

### Elementos Obrigatórios
- [x] Capa
- [x] Folha de rosto
- [ ] Folha de aprovação (se aplicável)
- [x] Dedicatória (opcional)
- [x] Agradecimentos (opcional)
- [ ] Resumo (FALTANDO)
- [ ] Abstract (FALTANDO)
- [x] Lista de figuras
- [x] Lista de tabelas
- [x] Sumário
- [x] Introdução
- [x] Referencial Teórico
- [x] Materiais e Métodos
- [x] Resultados
- [ ] Conclusões (FALTANDO)
- [x] Referências
- [x] Apêndices (ok)
- [x] Anexos (ok)

### Verificações Técnicas
- [x] Todas as citações estão no BibTeX (66/66)
- [x] Todas as figuras existem (7/7 verificadas)
- [ ] Numeração de seções consistente
- [ ] Todas as tabelas têm fonte
- [ ] Todas as figuras têm legenda
- [ ] Siglas definidas na primeira aparição
- [ ] Equações numeradas corretamente

### Qualidade de Conteúdo
- [x] Objetivos claros e mensuráveis
- [x] Metodologia reprodutível
- [x] Resultados apresentados com clareza
- [ ] Discussão crítica dos resultados
- [ ] Limitações explicitadas
- [ ] Trabalhos futuros sugeridos

---

## ESTIMATIVA DE TEMPO

| Tarefa | Tempo Estimado | Prioridade |
|--------|----------------|------------|
| Escrever Conclusões | 4-6 horas | CRÍTICA |
| Escrever Resumo/Abstract | 2-3 horas | CRÍTICA |
| Alinhar Objetivos | 2-3 horas | CRÍTICA |
| Revisar Introdução | 2-3 horas | ALTA |
| Melhorar Resultados | 3-4 horas | ALTA |
| Verificar Consistência | 2 horas | ALTA |
| Adicionar Figuras RT | 3-4 horas | MÉDIA |
| Melhorar Tabelas | 1-2 horas | MÉDIA |
| Revisar Redação | 3-4 horas | MÉDIA |
| Apêndices/Anexos | 2-3 horas | MÉDIA |
| **TOTAL** | **24-34 horas** | |

**Recomendação**: Focar nas CRÍTICAS primeiro (8-12h), depois ALTAS (7-10h). Total mínimo: 15-22 horas de trabalho focado.

---

## COMENTÁRIOS FINAIS

### Pontos Muito Positivos
1. **Experimentos sólidos**: 295 execuções é um número robusto
2. **Variedade de técnicas**: 20 métodos diferentes mostra amplitude
3. **Análise quantitativa**: Múltiplas métricas bem escolhidas
4. **Dados reais**: Não são resultados simulados, são resultados reais
5. **Documentação detalhada**: Materiais e Métodos muito completo

### Principais Gaps
1. **Falta conclusão**: Sem conclusão o trabalho está incompleto
2. **Desalinhamento**: Objetivos não batem com resultados
3. **Textos pré-textuais vazios**: Resumo e Abstract faltando

### Potencial de Melhoria
Com as correções críticas, o TCC pode facilmente subir para **9.0/10**.
O trabalho experimental é muito bom, só precisa de fechamento e alinhamento.

### Recomendação de Prazo
- **Mínimo**: 1 semana (fazendo só o crítico + alto)
- **Ideal**: 2 semanas (incluindo melhorias médias)
- **Confortável**: 3 semanas (polimento completo)

---

## PRÓXIMOS PASSOS IMEDIATOS

1. **DIA 1-2**: Escrever capítulo de Conclusões (4-6h)
2. **DIA 2-3**: Escrever Resumo e Abstract (2-3h)
3. **DIA 3-4**: Revisar e alinhar Introdução/Objetivos (3-4h)
4. **DIA 5**: Melhorar capítulo de Resultados (3-4h)
5. **DIA 6**: Verificar consistência geral (2h)
6. **DIA 7**: Revisão final e compilação LaTeX

**Boa sorte com as melhorias! O trabalho tem uma base muito boa.**
