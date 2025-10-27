"""
Módulo de métricas para avaliação de qualidade de fusão de imagens
Autor: Sistema de Stitching Deep Learning
"""

import numpy as np
import cv2
from typing import Dict, Optional
import sys

# Tentar importar scikit-image (preferencial)
try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from skimage.measure import shannon_entropy
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("⚠️  scikit-image não encontrado. Usando fallback para métricas.")
    print("   Para melhor precisão: pip install scikit-image")


class MetricasImageStitching:
    """
    Classe para calcular todas as métricas de qualidade de fusão de imagens
    """

    def __init__(self):
        self.metricas = {}

    def calcular_psnr(self, ground_truth: np.ndarray, resultado: np.ndarray) -> float:
        """
        Calcula PSNR (Peak Signal-to-Noise Ratio) entre ground truth e resultado

        Args:
            ground_truth: Imagem de referência (H, W, 3) RGB
            resultado: Imagem resultado (H, W, 3) RGB

        Returns:
            PSNR em dB (decibéis)
        """
        # Garantir mesmas dimensões
        if ground_truth.shape != resultado.shape:
            print(f"⚠️  Redimensionando resultado de {resultado.shape[:2]} para {ground_truth.shape[:2]}")
            resultado = cv2.resize(resultado,
                                  (ground_truth.shape[1], ground_truth.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

        if HAS_SKIMAGE:
            return peak_signal_noise_ratio(ground_truth, resultado, data_range=255)
        else:
            # Fallback usando OpenCV
            # Converter RGB -> BGR para OpenCV
            gt_bgr = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2BGR)
            res_bgr = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
            return float(cv2.PSNR(gt_bgr, res_bgr))

    def calcular_ssim(self, ground_truth: np.ndarray, resultado: np.ndarray) -> float:
        """
        Calcula SSIM (Structural Similarity Index) entre ground truth e resultado

        Args:
            ground_truth: Imagem de referência (H, W, 3) RGB
            resultado: Imagem resultado (H, W, 3) RGB

        Returns:
            SSIM entre 0 e 1 (1 = idêntico)
        """
        # Garantir mesmas dimensões
        if ground_truth.shape != resultado.shape:
            resultado = cv2.resize(resultado,
                                  (ground_truth.shape[1], ground_truth.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

        if HAS_SKIMAGE:
            return structural_similarity(ground_truth, resultado,
                                        channel_axis=2,
                                        data_range=255)
        else:
            # Fallback: SSIM manual em grayscale
            gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY)
            res_gray = cv2.cvtColor(resultado, cv2.COLOR_RGB2GRAY)

            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            gt_f = gt_gray.astype(np.float32)
            res_f = res_gray.astype(np.float32)

            mu1 = cv2.GaussianBlur(gt_f, (7, 7), 1.5)
            mu2 = cv2.GaussianBlur(res_f, (7, 7), 1.5)

            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu12 = mu1 * mu2

            sigma1_sq = cv2.GaussianBlur(gt_f * gt_f, (7, 7), 1.5) - mu1_sq
            sigma2_sq = cv2.GaussianBlur(res_f * res_f, (7, 7), 1.5) - mu2_sq
            sigma12 = cv2.GaussianBlur(gt_f * res_f, (7, 7), 1.5) - mu12

            ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                      ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-12)

            return float(np.mean(ssim_map))

    def calcular_mse(self, ground_truth: np.ndarray, resultado: np.ndarray) -> float:
        """
        Calcula MSE (Mean Squared Error) entre ground truth e resultado

        Args:
            ground_truth: Imagem de referência
            resultado: Imagem resultado

        Returns:
            MSE (quanto menor, melhor)
        """
        if ground_truth.shape != resultado.shape:
            resultado = cv2.resize(resultado,
                                  (ground_truth.shape[1], ground_truth.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)

        return float(np.mean((ground_truth.astype(float) - resultado.astype(float)) ** 2))

    def calcular_entropia(self, imagem: np.ndarray) -> float:
        """
        Calcula entropia média da imagem (RGB)
        Entropia mede a quantidade de informação/complexidade na imagem

        Args:
            imagem: Imagem RGB (H, W, 3)

        Returns:
            Entropia média dos 3 canais
        """
        if HAS_SKIMAGE:
            entropias = [shannon_entropy(imagem[:, :, i]) for i in range(3)]
            return float(np.mean(entropias))
        else:
            # Fallback: entropia manual
            entropias = []
            for i in range(3):
                channel = imagem[:, :, i]
                hist, _ = np.histogram(channel, bins=256, range=(0, 256))
                hist = hist / hist.sum()  # Normalizar
                hist = hist[hist > 0]  # Remover zeros
                entropy = -np.sum(hist * np.log2(hist))
                entropias.append(entropy)
            return float(np.mean(entropias))

    def calcular_nitidez(self, imagem: np.ndarray) -> float:
        """
        Calcula nitidez usando variância do Laplaciano
        Detecta o quão nítida (vs borrada) está a imagem

        Args:
            imagem: Imagem RGB (H, W, 3)

        Returns:
            Valor de nitidez (maior = mais nítida)
            Tipicamente: > 100 = nítida, < 100 = borrada
        """
        gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def calcular_todas(self,
                      resultado: np.ndarray,
                      ground_truth: Optional[np.ndarray] = None,
                      tempo: float = 0.0) -> Dict[str, float]:
        """
        Calcula todas as métricas possíveis

        Args:
            resultado: Imagem resultado da fusão (RGB)
            ground_truth: Imagem de referência (opcional)
            tempo: Tempo de processamento em segundos

        Returns:
            Dicionário com todas as métricas:
            - tempo_segundos: tempo de execução
            - psnr_db: PSNR em dB (ou None se sem ground truth)
            - ssim: SSIM 0-1 (ou None se sem ground truth)
            - mse: MSE (ou None se sem ground truth)
            - entropia: Entropia média
            - nitidez: Nitidez (variância Laplaciano)
        """
        metricas = {
            'tempo_segundos': tempo,
            'entropia': self.calcular_entropia(resultado),
            'nitidez': self.calcular_nitidez(resultado)
        }

        # Métricas que precisam de ground truth
        if ground_truth is not None:
            try:
                metricas['psnr_db'] = self.calcular_psnr(ground_truth, resultado)
                metricas['ssim'] = self.calcular_ssim(ground_truth, resultado)
                metricas['mse'] = self.calcular_mse(ground_truth, resultado)
            except Exception as e:
                print(f"⚠️  Erro ao calcular métricas com ground truth: {e}")
                metricas['psnr_db'] = None
                metricas['ssim'] = None
                metricas['mse'] = None
        else:
            metricas['psnr_db'] = None
            metricas['ssim'] = None
            metricas['mse'] = None

        return metricas


# ===== TESTE STANDALONE =====
if __name__ == "__main__":
    print("="*80)
    print("TESTE DO MÓDULO DE MÉTRICAS")
    print("="*80)

    # Verificar dependências
    print("\n📦 Dependências:")
    print(f"  - scikit-image: {'✅ OK' if HAS_SKIMAGE else '❌ Não instalado (usando fallback)'}")
    print(f"  - OpenCV: ✅ OK")
    print(f"  - NumPy: ✅ OK")

    # Criar teste simples
    print("\n🧪 Teste 1: Imagens sintéticas")
    calc = MetricasImageStitching()

    # Criar imagem ground truth (degradê)
    ground_truth = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        ground_truth[i, :, :] = i * 255 // 100

    # Criar resultado idêntico (deve ter PSNR alto, SSIM ~1)
    resultado_perfeito = ground_truth.copy()

    # Criar resultado com ruído (deve ter PSNR menor, SSIM menor)
    resultado_ruido = ground_truth.copy()
    ruido = np.random.randint(-20, 20, ground_truth.shape, dtype=np.int16)
    resultado_ruido = np.clip(resultado_ruido.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    print("\n📊 Resultado Perfeito (cópia do ground truth):")
    metricas_perfeito = calc.calcular_todas(resultado_perfeito, ground_truth, tempo=0.1)
    for k, v in metricas_perfeito.items():
        if v is not None:
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print("\n📊 Resultado com Ruído:")
    metricas_ruido = calc.calcular_todas(resultado_ruido, ground_truth, tempo=0.2)
    for k, v in metricas_ruido.items():
        if v is not None:
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print("\n✅ Teste concluído!")
    print("\nPara usar com imagens reais:")
    print("  1. Importe: from metricas import MetricasImageStitching")
    print("  2. Crie instância: calc = MetricasImageStitching()")
    print("  3. Calcule: metricas = calc.calcular_todas(resultado, ground_truth, tempo)")
    print("="*80)
