"""
Directional Stitcher
Realiza stitching com direção específica (horizontal ou vertical)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class DirectionalStitcher:
    """
    Stitcher com controle de direção (horizontal ou vertical)
    """

    def __init__(self, direction: str = 'horizontal', base_method: Optional[any] = None):
        """
        Inicializa o stitcher direcional

        Args:
            direction: 'horizontal' ou 'vertical'
            base_method: Método base de stitching (ex: método clássico ou deep)
                        Se None, usa cv2.Stitcher
        """
        if direction not in ['horizontal', 'vertical']:
            raise ValueError(f"Direção inválida: {direction}. Use 'horizontal' ou 'vertical'")

        self.direction = direction
        self.base_method = base_method

    def stitch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """
        Realiza stitching das imagens na direção especificada

        Args:
            images: Lista de imagens

        Returns:
            Tupla (resultado, info)
        """
        if len(images) < 2:
            raise ValueError("Necessário pelo menos 2 imagens para stitching")

        # Ordenar imagens baseado na direção
        images = self._order_images(images)

        # Se tem método base personalizado, usa ele
        if self.base_method is not None:
            return self._stitch_with_custom_method(images)

        # Caso contrário, usa OpenCV Stitcher com configurações direcionais
        return self._stitch_with_opencv(images)

    def _order_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Ordena imagens baseado na direção

        Para horizontal: esquerda -> direita
        Para vertical: cima -> baixo
        """
        # Por enquanto retorna as imagens na ordem original
        # Pode ser melhorado com análise de features para ordenação automática
        return images

    def _stitch_with_opencv(self, images: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """Usa OpenCV Stitcher"""
        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)

            # Configurar para melhor performance em direção específica
            if self.direction == 'horizontal':
                # Configurações para panorama horizontal
                pass  # Configurações padrão já funcionam bem
            else:  # vertical
                # Para vertical, pode precisar rotacionar imagens
                rotated = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in images]
                status, pano = stitcher.stitch(rotated)

                if status == cv2.Stitcher_OK:
                    # Rotacionar resultado de volta
                    pano = cv2.rotate(pano, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    return pano, {'status': 'success', 'direction': self.direction}
                else:
                    return None, {'status': 'error', 'code': status}

            status, pano = stitcher.stitch(images)

            if status == cv2.Stitcher_OK:
                return pano, {'status': 'success', 'direction': self.direction}
            else:
                return None, {'status': 'error', 'code': status}

        except Exception as e:
            return None, {'status': 'error', 'message': str(e)}

    def _stitch_with_custom_method(self, images: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """Usa método customizado"""
        try:
            # Aplicar método base
            if hasattr(self.base_method, 'stitch'):
                result = self.base_method.stitch(images)
            elif callable(self.base_method):
                result = self.base_method(images)
            else:
                raise ValueError("Método base inválido")

            if result is not None:
                return result, {'status': 'success', 'direction': self.direction, 'method': 'custom'}
            else:
                return None, {'status': 'error', 'message': 'Método customizado falhou'}

        except Exception as e:
            return None, {'status': 'error', 'message': str(e)}

    def stitch_horizontal(self, images: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """Atalho para stitching horizontal"""
        original_direction = self.direction
        self.direction = 'horizontal'
        result = self.stitch(images)
        self.direction = original_direction
        return result

    def stitch_vertical(self, images: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """Atalho para stitching vertical"""
        original_direction = self.direction
        self.direction = 'vertical'
        result = self.stitch(images)
        self.direction = original_direction
        return result


def stitch_images_directional(images: List[np.ndarray],
                              direction: str = 'horizontal',
                              method: Optional[any] = None) -> Tuple[Optional[np.ndarray], dict]:
    """
    Função auxiliar para stitching direcional

    Args:
        images: Lista de imagens
        direction: 'horizontal' ou 'vertical'
        method: Método de stitching (opcional)

    Returns:
        Tupla (resultado, info)
    """
    stitcher = DirectionalStitcher(direction=direction, base_method=method)
    return stitcher.stitch(images)
