# 🚀 Guia Rápido - Image Stitcher Pro

## ⚡ Início Rápido (2 minutos)

### 1️⃣ Instalar Dependências
```bash
pip install opencv-python opencv-contrib-python numpy pillow
```

### 2️⃣ Executar GUI
```bash
python tk_stitcher_gui.py
```

### 3️⃣ Fazer Primeiro Stitch
1. Clique em **"Adicionar imagens..."** ou pressione `Ctrl+O`
2. Selecione 2+ imagens com sobreposição
3. Clique em **"▶ Executar"** ou pressione `Ctrl+R`
4. Aguarde! Resultado abrirá automaticamente

**Pronto!** 🎉 Seu primeiro panorama foi criado.

---

## 🎯 Cenários Comuns

### Panorama de Paisagem
```
1. Adicionar 3-5 fotos da esquerda para direita
2. Método: "Deep • LoFTR" ou "Clássico • Seams Multiband"
3. Modo: Panorama
4. Executar
```

### Digitalizar Documento Grande
```
1. Adicionar fotos do documento em ordem
2. Método: "Clássico • OpenCV Stitcher"
3. Modo: Scans
4. Executar
```

### Comparar Todos os Métodos
```
1. Adicionar imagens
2. Pressionar F5 (ou clicar "▶▶ Todos")
3. Aguardar conclusão
4. Abrir pasta results/all_XXXXXX
5. Comparar resultados visualmente
```

### Trabalho Repetitivo
```
1. Configurar método, modo, GPU, etc.
2. Ctrl+S → Digitar nome do preset → OK
3. Próxima vez: Ctrl+P → Escolher preset → OK
4. Executar!
```

---

## 💡 Dicas Essenciais

### ✅ DO (Faça)
- ✅ Use 30-50% de sobreposição entre imagens
- ✅ Mantenha iluminação consistente
- ✅ Tire fotos na mesma altura/ângulo
- ✅ Ordene imagens antes de executar
- ✅ Use modo **Panorama** para paisagens
- ✅ Use modo **Scans** para documentos planos

### ❌ DON'T (Evite)
- ❌ Imagens sem overlap
- ❌ Mudanças drásticas de luz entre fotos
- ❌ Objetos em movimento no overlap
- ❌ Ordem aleatória de imagens
- ❌ Zoom/foco diferentes entre imagens

---

## 🔥 Atalhos Mais Usados

| Atalho | Ação |
|--------|------|
| `Ctrl+O` | Adicionar imagens |
| `Ctrl+R` | Executar stitch |
| `F5` | Executar TODOS os métodos |
| `Ctrl+S` | Salvar preset |
| `Ctrl+P` | Carregar preset |
| `Ctrl+H` | Ver histórico |
| `Ctrl+T` | Trocar tema Dark/Light |
| `F1` | Ajuda completa |
| `Delete` | Remover imagens selecionadas |

---

## 🏆 Melhores Métodos por Caso

| Caso de Uso | Método Recomendado | Por quê? |
|-------------|-------------------|----------|
| **Paisagens** | LoFTR ou Seams Multiband | Alta qualidade, bom blend |
| **Documentos** | OpenCV Stitcher | Rápido, preciso para planos |
| **Baixa textura** | LoFTR ou ORB | Detector-free funciona melhor |
| **Velocidade** | SIFT Básico | Mais rápido, resultado ok |
| **Qualidade máxima** | Seams Multiband | Graph-cut + multiband = pro |
| **Fotos noturnas** | Blend Exposure | Equaliza iluminação |
| **Macro/close-up** | ECC Alignment | Alinhamento preciso |

---

## 📊 Interpretar Resultados

### Quando Funciona Bem
- ✅ Sem costuras visíveis
- ✅ Cores consistentes
- ✅ Sem fantasmas (ghosting)
- ✅ Proporções corretas

### Problemas Comuns

#### 🔴 Costuras Visíveis
**Solução**: Tente métodos com blend avançado:
- Seams Multiband
- Blend Laplacian
- Blend Poisson

#### 🔴 Fantasmas/Duplicação
**Causa**: Objetos em movimento no overlap
**Solução**:
- Retake fotos sem movimento
- Tente método com melhor detecção (LoFTR)

#### 🔴 Cores Diferentes
**Solução**: Use métodos com compensação:
- Exposure Comp Only
- Blend Exposure
- Blend Hybrid

#### 🔴 Falha Total
**Causas possíveis**:
- Overlap insuficiente
- Imagens muito diferentes
- Ordem errada

**Solução**:
1. Verifique overlap (30-50% mínimo)
2. Ordene imagens corretamente
3. Tente método mais robusto (LoFTR)
4. Use F5 para comparar todos

---

## 🎨 Recursos Avançados

### Histórico de Execuções
```
Ctrl+H → Ver todas execuções
         Tempo, método, status
         Exportar CSV para análise
```

### Relatórios Batch
```
F5 → Executar todos
Ctrl+E → Gerar relatório HTML
         Abre automaticamente
         Mostra todos resultados
```

### Presets para Projetos
```
Preset "Outdoor"
- Método: LoFTR
- Modo: Panorama
- GPU: ✓

Preset "Docs"
- Método: OpenCV Stitcher
- Modo: Scans
- GPU: ✗

Preset "Quality"
- Método: Seams Multiband
- Modo: Panorama
- Métricas: ✓
```

### Tema Dark
```
Ctrl+T para alternar
Melhor para:
- Trabalho noturno
- Longas sessões
- Reduzir fadiga ocular
```

---

## 🔧 Troubleshooting Rápido

### "Não há imagens suficientes"
➜ Adicione pelo menos 2 imagens

### "Falha ao estimar homografia"
➜ Aumente overlap ou tente outro método

### "Timeout excedido"
➜ Normal para métodos deep learning
➜ Tente método clássico (mais rápido)

### Método deep falha
➜ Instale PyTorch: `pip install torch torchvision`

### Preview não aparece
➜ Instale Pillow: `pip install pillow`

### Drag & drop não funciona
➜ Opcional: `pip install tkinterdnd2`

---

## 📚 Quer Saber Mais?

- **README_GUI.md** → Documentação completa
- **CHANGELOG_GUI.md** → Todas as funcionalidades
- **F1 na GUI** → Ajuda rápida com atalhos

---

## 🎓 Fluxo Profissional

```
1. PREPARAR
   ├─ Tirar fotos com 30-50% overlap
   ├─ Mesma altura/ângulo
   └─ Iluminação consistente

2. CARREGAR
   ├─ Ctrl+O ou drag & drop
   ├─ Ordenar com ▲/▼
   └─ Verificar preview

3. CONFIGURAR
   ├─ Escolher método (ver descrição na status bar)
   ├─ Modo: Panorama ou Scans
   └─ Carregar preset se já tiver (Ctrl+P)

4. EXECUTAR
   ├─ Ctrl+R para um método
   └─ F5 para comparar todos

5. ANALISAR
   ├─ Abrir resultado (🖼 botão)
   ├─ Verificar qualidade
   └─ Ver histórico (Ctrl+H)

6. OTIMIZAR (se necessário)
   ├─ Testar outro método
   ├─ Ajustar ordem das imagens
   └─ Retirar/adicionar fotos

7. EXPORTAR
   ├─ Batch: Ctrl+E para relatório
   └─ Salvar configuração: Ctrl+S
```

---

**Tudo Pronto!** 🚀

Comece com o fluxo básico e explore recursos avançados conforme necessário.

**Boa sorte com seus panoramas!** 📸✨
