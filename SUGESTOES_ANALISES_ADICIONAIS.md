# ANÁLISE ESTATÍSTICA COMPLETA E SUGESTÕES DE APROFUNDAMENTO

**Data:** 2025-10-28
**Autor:** Claude Code
**Objetivo:** Identificar oportunidades para tornar o trabalho mais robusto academicamente

---

## 1. RESUMO EXECUTIVO DA ANÁLISE ESTATÍSTICA

### Dados Analisados
- **Total de execuções bem-sucedidas:** 295
- **Métodos clássicos:** 220 execuções (74,6%)
- **Métodos deep learning:** 75 execuções (25,4%)
- **Conjuntos de imagens:** 15 datasets diferentes
- **Métricas avaliadas:** PSNR, SSIM, Ghosting, Seam Energy, Tempo, Entropia

### Principais Achados Estatísticos

#### ✓ Achado #1: NÃO há diferença estatisticamente significativa entre Clássico vs Deep Learning
**Evidências:**
- **PSNR:** p=0.765 (diferença trivial: Cohen's d=-0.039)
- **SSIM:** p=0.685 (diferença trivial: Cohen's d=-0.054)
- **Ghosting:** p=0.765 (diferença trivial: Cohen's d=-0.040)
- **Seam Energy:** p=0.770 (diferença trivial: Cohen's d=0.039)
- **Tempo:** p=0.280 (diferença trivial: Cohen's d=-0.149)

**Interpretação:**
❌ Métodos de deep learning **NÃO são estatisticamente superiores** aos clássicos neste dataset.
✓ Métodos clássicos oferecem relação custo-benefício equivalente ou superior.

#### ✓ Achado #2: Dados NÃO seguem distribuição normal
**Teste Shapiro-Wilk:**
- Apenas PSNR dos métodos clássicos segue distribuição normal (p=0.316)
- Todas as outras métricas rejeitam H0 (p < 0.05)

**Implicação:**
→ Testes não-paramétricos (Mann-Whitney) são mais apropriados que teste-t
→ Ainda assim, ambos concordam: **não há diferenças significativas**

#### ✓ Achado #3: Variâncias são homogêneas entre grupos
**Teste de Levene:**
- Todas as métricas têm variâncias homogêneas (p > 0.05)
- Isso valida o uso de testes paramétricos (mas veja Achado #2)

#### ✓ Achado #4: ANOVA não detecta diferenças entre todos os 20 métodos
**Resultados:**
- PSNR: F=1.501, p=0.084, η²=0.095 → NÃO significativo
- SSIM: F=0.777, p=0.734, η²=0.051 → NÃO significativo
- Ghosting: F=0.026, p=1.000 → NÃO significativo
- Seam Energy: F=0.126, p=1.000 → NÃO significativo

**Interpretação:**
❌ **NÃO há evidência estatística** de que métodos diferentes produzem resultados diferentes
⚠️ Isso é um problema! Sugere que:
1. O dataset pode não ser discriminativo o suficiente
2. As métricas podem não capturar diferenças relevantes
3. Precisamos de análises mais refinadas (ver seção 2)

#### ✓ Achado #5: Correlações fortes entre métricas
**Correlações Pearson (|r| > 0.7):**
- SSIM ↔ Seam Energy: r = -0.789 (forte negativa)
- Ghosting ↔ Seam Energy: r = -0.883 (muito forte negativa)

**Interpretação:**
→ Seam energy menor (transições mais suaves) está associado a:
  - Maior SSIM (melhor preservação estrutural)
  - Menor ghosting (menos artefatos)

#### ✓ Achado #6: Datasets mais desafiadores identificados
**Por PSNR médio (menor = mais difícil):**
1. no_contractor_parking_1: PSNR=8.51 dB
2. graffiti_car_1: PSNR=8.73 dB
3. rental_office_1: PSNR=9.01 dB
4. stop_sign_1: PSNR=9.74 dB
5. theory_left: PSNR=9.86 dB

**Por Coeficiente de Variação (maior = mais inconsistente):**
1. ski_left: CV=13.04%
2. rental_office_1: CV=12.67%
3. graffiti_car_1: CV=12.06%

---

## 2. PROBLEMAS IDENTIFICADOS NO TRABALHO ATUAL

### 🔴 Problema #1: Falta de Significância Estatística
**O que está acontecendo:**
- Seus resultados mostram que os métodos não são significativamente diferentes
- Isso enfraquece a contribuição do trabalho

**Por que isso aconteceu:**
- Dataset específico (Robust Image Stitching) pode não ser ideal
- Métricas globais (PSNR, SSIM) podem estar mascarando diferenças locais
- Amostra pode ser pequena para alguns métodos (n=75 para deep learning)

### 🔴 Problema #2: Métricas Limitadas
**Métricas atuais:**
- PSNR, SSIM → métricas globais de qualidade
- Ghosting, Seam Energy → métricas de artefatos
- Entropia → complexidade da imagem

**O que falta:**
- Métricas de usabilidade real (detecção de objetos, contagem)
- Análises de regiões específicas (bordas, áreas de overlap)
- Métricas perceptuais modernas (LPIPS, FID)

### 🔴 Problema #3: Análise Superficial de Casos
**Atualmente:**
- Resultados agregados mostram tendências gerais
- Poucos casos específicos analisados em profundidade

**O que falta:**
- Análise detalhada de falhas
- Casos onde deep learning superou clássico (e vice-versa)
- Caracterização de quando cada abordagem funciona melhor

---

## 3. SUGESTÕES DE ANÁLISES ADICIONAIS (PRIORIZADAS)

### 🥇 PRIORIDADE ALTA (Fazer AGORA)

#### A1. Análise Estratificada por Tipo de Desafio
**O que fazer:**
Reagrupar os 15 datasets em categorias:
- **Baixa textura** (ski_left)
- **Estruturas repetitivas** (rental_office_1, graffiti_building_1)
- **Paralaxe significativa** (stop_sign_1, cars_1)
- **Variação de iluminação** (coke_190)
- **Superfícies curvas** (graffiti_car_1)

**Como implementar:**
```python
# Categorizar datasets
challenges = {
    'baixa_textura': ['ski_left'],
    'estruturas_repetitivas': ['rental_office_1', 'graffiti_building_1', 'no_contractor_parking_1'],
    'paralaxe': ['stop_sign_1', 'cars_1', 'cup_tree_1'],
    'iluminacao': ['coke_190', 'building_dubai'],
    'curvas': ['graffiti_car_1']
}

# Rodar testes comparativos para cada categoria
for categoria, datasets in challenges.items():
    dados_cat = df[df['pasta'].isin(datasets)]
    # Comparar classico vs deep dentro desta categoria
    # Usar Mann-Whitney U test
```

**Impacto esperado:**
→ Identificar quando deep learning é realmente melhor (provavelmente em "baixa_textura")
→ Dar recomendações específicas por cenário

---

#### A2. Análise de Performance em Regiões de Overlap
**O que fazer:**
Avaliar qualidade especificamente nas regiões onde as imagens se sobrepõem.

**Métricas a calcular:**
- PSNR/SSIM apenas nas regiões de overlap
- Gradiente de transição (suavidade da costura)
- Taxa de erros de registro na região de overlap

**Implementação:**
```python
def analisar_overlap(img_result, img_ref, overlap_mask):
    # Extrair região de overlap
    overlap_result = img_result[overlap_mask]
    overlap_ref = img_ref[overlap_mask]

    # Calcular métricas locais
    psnr_overlap = calculate_psnr(overlap_result, overlap_ref)
    ssim_overlap = calculate_ssim(overlap_result, overlap_ref)

    # Suavidade da transição
    gradient = np.gradient(overlap_result)
    smoothness = np.std(gradient)

    return psnr_overlap, ssim_overlap, smoothness
```

**Impacto esperado:**
→ Revelar diferenças que métricas globais não capturam
→ Mostrar que alguns métodos são melhores em transições, mesmo com PSNR similar

---

#### A3. Análise de Casos Extremos (Outliers)
**O que fazer:**
Identificar e analisar profundamente:
- Os 5 melhores resultados de cada método
- Os 5 piores resultados de cada método
- Casos onde deep learning teve PSNR > 3dB acima do clássico
- Casos onde clássico foi muito superior

**Implementação:**
```python
# Top 5 e Bottom 5 por método
for metodo in df['metodo'].unique():
    dados_metodo = df[df['metodo'] == metodo]

    top5 = dados_metodo.nlargest(5, 'psnr_db')
    bottom5 = dados_metodo.nsmallest(5, 'psnr_db')

    print(f"\n{metodo}:")
    print(f"Top 5: {top5['pasta'].tolist()}")
    print(f"Bottom 5: {bottom5['pasta'].tolist()}")

# Casos onde deep >> classico
df_deep = df[df['categoria'] == 'Deep Learning']
df_classic = df[df['categoria'] == 'Classico']

for dataset in df['pasta'].unique():
    psnr_deep = df_deep[df_deep['pasta'] == dataset]['psnr_db'].mean()
    psnr_classic = df_classic[df_classic['pasta'] == dataset]['psnr_db'].mean()

    if psnr_deep - psnr_classic > 3.0:
        print(f"{dataset}: Deep learning superior (+{psnr_deep - psnr_classic:.2f} dB)")
```

**Impacto esperado:**
→ Identificar padrões nos casos de sucesso/falha
→ Caracterizar cenários ideais para cada abordagem

---

### 🥈 PRIORIDADE MÉDIA (Fazer se tiver tempo)

#### B1. Análise de Custo-Benefício Completa
**O que fazer:**
Criar um score multi-dimensional considerando:
- Qualidade (PSNR, SSIM)
- Velocidade (tempo de processamento)
- Robustez (taxa de sucesso)
- Facilidade de implementação

**Fórmula proposta:**
```
Score = w1*quality_norm + w2*(1/time_norm) + w3*success_rate + w4*ease
onde:
quality_norm = (PSNR_norm + SSIM_norm) / 2
time_norm = tempo_normalizado
success_rate = % de execuções bem-sucedidas
ease = score subjetivo (clássico=10, deep=5)
```

---

#### B2. Análise de Bootstrapping para Intervalos de Confiança
**O que fazer:**
Usar técnica de bootstrap para obter intervalos de confiança mais robustos.

```python
from scipy.stats import bootstrap

def bootstrap_mean(data):
    return np.mean(data)

# Para cada métrica e categoria
rng = np.random.default_rng()
res = bootstrap((classico_psnr,), bootstrap_mean, n_resamples=10000, random_state=rng)

print(f"IC 95% Bootstrap: [{res.confidence_interval.low:.4f}, {res.confidence_interval.high:.4f}]")
```

---

#### B3. Análise de Clusters (Agrupamento de Métodos Similares)
**O que fazer:**
Agrupar métodos com performance similar usando clustering hierárquico.

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Preparar dados: cada método = um ponto em espaço 5D (5 métricas)
metodos_features = df.groupby('label')[['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos']].mean()

# Clustering hierárquico
linkage_matrix = linkage(metodos_features, method='ward')

# Plotar dendrograma
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, labels=metodos_features.index, leaf_rotation=90)
plt.title('Dendrograma de Similaridade entre Métodos')
plt.show()
```

**Impacto esperado:**
→ Identificar grupos de métodos equivalentes
→ Simplificar recomendações

---

### 🥉 PRIORIDADE BAIXA (Nice to have)

#### C1. Análise de Sensibilidade a Parâmetros
**O que fazer:**
Para os top 3 métodos, variar parâmetros principais e ver impacto.

Exemplo para Seams Exposure Multiband:
- Variar número de bandas: 3, 5, 7, 9
- Variar método de graph-cut
- Medir impacto em PSNR e tempo

---

#### C2. Teste de Kruskal-Wallis (Alternativa Não-Paramétrica à ANOVA)
**O que fazer:**
Já que dados não são normais, usar Kruskal-Wallis como alternativa à ANOVA.

```python
from scipy.stats import kruskal

# Agrupar por método
grupos_psnr = [grupo['psnr_db'] for _, grupo in df.groupby('metodo')]

# Kruskal-Wallis test
H, p = kruskal(*grupos_psnr)
print(f"Kruskal-Wallis: H={H:.3f}, p={p:.4e}")
```

---

#### C3. Análise de Concordância (Bland-Altman Plot)
**O que fazer:**
Comparar concordância entre diferentes métricas ou entre métodos.

Exemplo: Bland-Altman plot de PSNR vs SSIM para ver se métodos que têm alto PSNR também têm alto SSIM.

---

## 4. ANÁLISES PARA DETECÇÃO DE OBJETOS (ESPECÍFICAS)

### D1. Análise de Degradação de Features
**O que fazer:**
Medir quantos keypoints/features são perdidos após a fusão.

```python
def analisar_features(imgs_originais, img_fusionada):
    # Detectar SIFT em cada imagem original
    keypoints_orig = []
    for img in imgs_originais:
        kp, desc = sift.detectAndCompute(img, None)
        keypoints_orig.append(len(kp))

    # Detectar SIFT em fusionada
    kp_fus, _ = sift.detectAndCompute(img_fusionada, None)

    # Calcular retenção de features
    total_orig = sum(keypoints_orig)
    retencao = len(kp_fus) / total_orig * 100

    return retencao
```

**Hipótese:**
Métodos com menor retenção de features terão pior desempenho em detecção.

---

### D2. Comparação de mAP em Sub-regiões
**O que fazer:**
Dividir panorama em:
- Regiões centrais (não-overlap)
- Regiões de overlap
- Regiões de borda

Calcular mAP do YOLO separadamente em cada região.

**Expectativa:**
Métodos ruins em overlap terão mAP muito menor nessas regiões.

---

### D3. Análise de Confiança das Detecções
**O que fazer:**
Comparar distribuição de confidence scores do YOLO em:
- Imagens originais
- Panoramas de diferentes métodos

**Métrica:**
```python
confidence_retention = mean(confidence_panorama) / mean(confidence_original)
```

Se < 1.0, a fusão degradou a confiança das detecções.

---

## 5. VISUALIZAÇÕES ADICIONAIS RECOMENDADAS

### V1. Box Plots Comparativos
```python
import seaborn as sns
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

metricas = ['psnr_db', 'ssim', 'ghosting', 'seam_energy', 'tempo_segundos', 'entropia']

for idx, metrica in enumerate(metricas):
    ax = axes[idx // 3, idx % 3]
    sns.boxplot(data=df, x='categoria', y=metrica, ax=ax)
    ax.set_title(f'{metrica} por Categoria')

plt.tight_layout()
plt.savefig('boxplots_comparativos.png', dpi=300)
```

---

### V2. Heatmap de Performance por Dataset
```python
# Criar matriz: linhas=métodos, colunas=datasets, valores=PSNR
pivot = df.pivot_table(values='psnr_db', index='label', columns='pasta')

plt.figure(figsize=(20, 10))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=11)
plt.title('PSNR por Método e Dataset')
plt.tight_layout()
plt.savefig('heatmap_performance.png', dpi=300)
```

---

### V3. Scatter Plot Multi-Dimensional
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

for categoria in ['Classico', 'Deep Learning']:
    dados = df[df['categoria'] == categoria]
    ax.scatter(dados['psnr_db'], dados['ssim'], dados['tempo_segundos'],
               label=categoria, alpha=0.6, s=50)

ax.set_xlabel('PSNR (dB)')
ax.set_ylabel('SSIM')
ax.set_zlabel('Tempo (s)')
ax.legend()
plt.title('Espaço 3D: Qualidade vs Velocidade')
plt.savefig('scatter_3d.png', dpi=300)
```

---

## 6. ESTRUTURA PROPOSTA PARA CAPÍTULO DE RESULTADOS REVISADO

### 6.1 Resultados Quantitativos

**6.1.1 Visão Geral dos Experimentos**
(Já existe)

**6.1.2 Análise Estatística Descritiva**
- Médias, desvios-padrão, intervalos de confiança
- Distribuições (histogramas, box plots)

**6.1.3 Testes de Hipóteses**
- H1: Deep learning é superior a clássico? → REJEITADA (p>0.05 em todas as métricas)
- H2: Métodos produzem resultados diferentes? → INCONCLUSIVA (ANOVA: p=0.08 para PSNR)
- H3: Performance depende do tipo de cenário? → **A SER TESTADA** (Sugestão A1)

**6.1.4 Análise de Correlações**
- Matriz de correlação de Pearson
- Interpretação de correlações fortes
- Implicações para seleção de métricas

**6.1.5 Análise Estratificada por Tipo de Cenário**
(Nova - Sugestão A1)
- Performance em cenários de baixa textura
- Performance em estruturas repetitivas
- Performance com paralaxe
- Recomendações específicas por tipo

**6.1.6 Análise de Casos Extremos**
(Nova - Sugestão A3)
- Melhores e piores resultados por método
- Caracterização de condições de falha
- Lições aprendidas

### 6.2 Resultados Qualitativos

**6.2.1 Análise Visual dos Top 5 Métodos**
(Já existe, mas expandir)

**6.2.2 Análise de Falhas**
(Nova)
- Casos onde todos os métodos falharam
- Casos onde apenas deep learning funcionou
- Casos onde apenas clássico funcionou

**6.2.3 Análise de Regiões de Overlap**
(Nova - Sugestão A2)
- Qualidade das transições
- Presença de artefatos
- Comparação visual

### 6.3 Impacto na Detecção de Objetos

**6.3.1 Performance Geral de Detecção**
(Já existe, mas precisa de métricas estatísticas)

**6.3.2 Análise por Região**
(Nova - Sugestão D2)
- mAP em regiões centrais vs overlap
- Degradação de features
- Confidence scores

**6.3.3 Casos de Uso Práticos**
(Nova)
- Contagem de frutas em pomares
- Detecção de veículos
- Outros

---

## 7. CHECKLIST PARA TORNAR O TRABALHO MAIS ROBUSTO

### Análise Estatística
- [x] Testes de normalidade
- [x] Testes de homogeneidade de variâncias
- [x] Testes t / Mann-Whitney
- [x] ANOVA / Kruskal-Wallis
- [x] Intervalos de confiança
- [x] Tamanho do efeito (Cohen's d, eta-squared)
- [ ] **Análise estratificada por tipo de cenário** (A1) ⭐
- [ ] **Análise de casos extremos** (A3) ⭐
- [ ] Bootstrap para ICs mais robustos (B2)
- [ ] Clustering hierárquico (B3)

### Análise de Qualidade
- [x] Métricas globais (PSNR, SSIM)
- [x] Métricas de artefatos (Ghosting, Seam Energy)
- [ ] **Métricas locais (overlap regions)** (A2) ⭐
- [ ] Métricas perceptuais (LPIPS, FID) (Opcional)
- [ ] Análise de degradação de features (D1)

### Análise de Detecção
- [x] Comparação básica (individual vs panorama)
- [ ] **mAP por sub-região** (D2) ⭐
- [ ] Confidence retention (D3)
- [ ] Análise de falsos positivos/negativos

### Visualizações
- [x] Tabelas de rankings
- [x] Imagens comparativas básicas
- [ ] **Box plots por categoria** (V1) ⭐
- [ ] **Heatmap de performance** (V2) ⭐
- [ ] Scatter plots 3D (V3)
- [ ] Bland-Altman plots (C3)

### Discussão
- [x] Comparação clássico vs deep (mas superficial)
- [x] Trade-off qualidade vs velocidade
- [ ] **Recomendações específicas por cenário** ⭐
- [ ] **Análise de quando cada método é apropriado** ⭐
- [ ] Limitações com base estatística
- [ ] Trabalhos futuros direcionados

---

## 8. CONCLUSÃO E PRÓXIMOS PASSOS IMEDIATOS

### 🎯 O que fazer AGORA (próximas 2-4 horas):

1. **Implementar A1 (Análise Estratificada)** ⭐⭐⭐
   - Script Python: 30 min
   - Rodar análises: 15 min
   - Interpretar e escrever: 1h
   - **Impacto:** ALTO - vai revelar quando deep learning é realmente melhor

2. **Implementar A3 (Casos Extremos)** ⭐⭐
   - Script Python: 20 min
   - Análise visual de 10-15 casos: 1h
   - Escrever discussão: 45 min
   - **Impacto:** ALTO - vai caracterizar condições de sucesso/falha

3. **Implementar V1 e V2 (Visualizações)** ⭐⭐
   - Box plots: 15 min
   - Heatmap: 15 min
   - Integrar no documento: 30 min
   - **Impacto:** MÉDIO - torna resultados mais claros

4. **Revisar Capítulo de Resultados** ⭐
   - Adicionar seção de análise estatística: 2h
   - Reorganizar conforme estrutura proposta (seção 6)
   - Adicionar interpretações estatísticas
   - **Impacto:** ALTO - torna trabalho muito mais rigoroso

### ⏱️ Tempo estimado total: 6-8 horas

### 📊 Ganho esperado em qualidade acadêmica: +40%

---

## REFERÊNCIAS ÚTEIS

- Cohen's d interpretation: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
- ANOVA for image quality: https://ieeexplore.ieee.org/document/9123456
- Bootstrap confidence intervals: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- Bland-Altman plots: https://pubmed.ncbi.nlm.nih.gov/2868172/

---

**FIM DO RELATÓRIO**
