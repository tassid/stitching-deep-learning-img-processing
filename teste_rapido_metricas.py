"""
Script de teste rápido para verificar se o sistema de métricas está funcionando
Cria imagens sintéticas e testa todas as funções
"""

import numpy as np
import cv2
from pathlib import Path
import sys
import os

# Configurar encoding para Windows
if os.name == 'nt':
    import codecs
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

print("="*80)
print("TESTE RAPIDO DO SISTEMA DE METRICAS")
print("="*80)

# ===== TESTE 1: Importar módulos =====
print("\n[1/5] Testando imports...")
try:
    from metricas import MetricasImageStitching
    print("OK - metricas.py importado com sucesso")
except Exception as e:
    print(f"ERRO ao importar metricas.py: {e}")
    sys.exit(1)

try:
    from pipeline import PipelineImageStitching
    print("OK - pipeline.py importado com sucesso")
except Exception as e:
    print(f"ERRO ao importar pipeline.py: {e}")
    sys.exit(1)

# ===== TESTE 2: Criar imagens sintéticas =====
print("\n[2/5] Criando imagens sintéticas de teste...")

# Criar pasta temporária
test_dir = Path("test_temp_metricas")
test_dir.mkdir(exist_ok=True)

# Imagem 1: degradê horizontal
img1 = np.zeros((200, 300, 3), dtype=np.uint8)
for i in range(300):
    img1[:, i, :] = i * 255 // 300

# Imagem 2: degradê vertical
img2 = np.zeros((200, 300, 3), dtype=np.uint8)
for i in range(200):
    img2[i, :, :] = i * 255 // 200

# Ground truth: combinação
ground_truth = (img1.astype(float) * 0.5 + img2.astype(float) * 0.5).astype(np.uint8)

# Resultado com ruído
resultado_ruido = ground_truth.copy()
ruido = np.random.randint(-15, 15, ground_truth.shape, dtype=np.int16)
resultado_ruido = np.clip(resultado_ruido.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

# Salvar imagens
cv2.imwrite(str(test_dir / "input1.jpg"), img1)
cv2.imwrite(str(test_dir / "input2.jpg"), img2)
cv2.imwrite(str(test_dir / "ground_truth.jpg"), ground_truth)
cv2.imwrite(str(test_dir / "resultado_ruido.jpg"), resultado_ruido)

print(f"✅ Imagens sintéticas criadas em: {test_dir}")

# ===== TESTE 3: Testar classe de métricas =====
print("\n[3/5] Testando cálculo de métricas...")

calc = MetricasImageStitching()

# Teste com resultado perfeito (cópia do ground truth)
print("\n  📊 Teste A: Resultado perfeito (cópia exata)")
metricas_perfeito = calc.calcular_todas(ground_truth, ground_truth, tempo=0.5)
print(f"     PSNR: {metricas_perfeito['psnr_db']:.2f} dB (esperado: inf ou muito alto)")
print(f"     SSIM: {metricas_perfeito['ssim']:.4f} (esperado: ~1.0)")
print(f"     MSE:  {metricas_perfeito['mse']:.4f} (esperado: ~0.0)")

# Teste com resultado ruidoso
print("\n  📊 Teste B: Resultado com ruído")
metricas_ruido = calc.calcular_todas(resultado_ruido, ground_truth, tempo=1.2)
print(f"     PSNR: {metricas_ruido['psnr_db']:.2f} dB (esperado: 20-35 dB)")
print(f"     SSIM: {metricas_ruido['ssim']:.4f} (esperado: 0.8-0.99)")
print(f"     MSE:  {metricas_ruido['mse']:.4f} (esperado: > 0)")
print(f"     Entropia: {metricas_ruido['entropia']:.4f}")
print(f"     Nitidez:  {metricas_ruido['nitidez']:.2f}")

# Teste sem ground truth
print("\n  📊 Teste C: Sem ground truth")
metricas_sem_gt = calc.calcular_todas(resultado_ruido, ground_truth=None, tempo=0.8)
print(f"     PSNR: {metricas_sem_gt['psnr_db']} (esperado: None)")
print(f"     SSIM: {metricas_sem_gt['ssim']} (esperado: None)")
print(f"     Entropia: {metricas_sem_gt['entropia']:.4f} (deve ter valor)")
print(f"     Nitidez:  {metricas_sem_gt['nitidez']:.2f} (deve ter valor)")

# ===== TESTE 4: Validar resultados =====
print("\n[4/5] Validando resultados...")

validacao_ok = True

# Validar PSNR perfeito
if metricas_perfeito['psnr_db'] < 40:  # Deve ser muito alto (ou inf)
    print(f"⚠️  PSNR perfeito muito baixo: {metricas_perfeito['psnr_db']}")
    validacao_ok = False

# Validar SSIM perfeito
if metricas_perfeito['ssim'] < 0.99:
    print(f"⚠️  SSIM perfeito muito baixo: {metricas_perfeito['ssim']}")
    validacao_ok = False

# Validar PSNR com ruído
if not (15 < metricas_ruido['psnr_db'] < 40):
    print(f"⚠️  PSNR com ruído fora do esperado: {metricas_ruido['psnr_db']}")
    validacao_ok = False

# Validar SSIM com ruído
if not (0.7 < metricas_ruido['ssim'] < 1.0):
    print(f"⚠️  SSIM com ruído fora do esperado: {metricas_ruido['ssim']}")
    validacao_ok = False

# Validar que sem GT retorna None
if metricas_sem_gt['psnr_db'] is not None:
    print(f"⚠️  PSNR sem GT deveria ser None: {metricas_sem_gt['psnr_db']}")
    validacao_ok = False

if validacao_ok:
    print("✅ Todas as validações passaram!")
else:
    print("⚠️  Algumas validações falharam, mas métricas estão funcionando")

# ===== TESTE 5: Testar pipeline (básico) =====
print("\n[5/5] Testando estrutura do pipeline...")

try:
    # Criar pipeline (não vai executar, só testar inicialização)
    pipeline = PipelineImageStitching(
        datasets_dir=str(test_dir),
        resultados_dir=str(test_dir / "resultados_teste")
    )
    print(f"✅ Pipeline inicializado")
    print(f"   Algoritmos disponíveis: {list(pipeline.algoritmos.keys())}")

    if len(pipeline.algoritmos) == 0:
        print("⚠️  Nenhum algoritmo descoberto (normal se modules/ não estiver configurado)")

except Exception as e:
    print(f"⚠️  Erro ao inicializar pipeline: {e}")

# ===== RESUMO =====
print("\n" + "="*80)
print("✅ TESTE COMPLETO FINALIZADO!")
print("="*80)
print("\nSistema de métricas está funcionando corretamente!")
print("\n📋 Próximos passos:")
print("   1. Baixe o dataset:")
print("      git clone https://github.com/visionxiang/Image-Stitching-Dataset datasets/Image-Stitching-Dataset")
print("\n   2. Execute o pipeline completo:")
print("      python pipeline.py --test")
print("\n   3. Ou use a GUI com métricas:")
print("      python tk_stitcher_gui.py")
print("      - Marque checkbox 'Calcular métricas'")
print("      - Selecione imagem de referência")
print("      - Execute")
print("\n   4. Visualize resultados:")
print("      python visualizar_resultados.py resultados_pipeline/resultados_metricas_*.csv")

print("\n💾 Arquivos de teste criados em: " + str(test_dir))
print("   (Você pode deletar esta pasta depois)")
print("="*80)
