"""
Grid Stitcher
Realiza stitching em formato de grid (2x2, 3x3, etc)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from .directional_stitcher import DirectionalStitcher


class GridStitcher:
    """
    Stitcher para múltiplas imagens em formato grid
    """

    def __init__(self, grid_size: Tuple[int, int] = (2, 2), base_method: Optional[any] = None):
        """
        Inicializa o grid stitcher

        Args:
            grid_size: Tupla (rows, cols) definindo o grid
            base_method: Método base de stitching
        """
        self.grid_size = grid_size
        self.base_method = base_method
        self.directional_stitcher = DirectionalStitcher(base_method=base_method)

    def stitch_grid(self, images: List[np.ndarray]) -> Tuple[Optional[np.ndarray], dict]:
        """
        Realiza stitching em grid

        Args:
            images: Lista de imagens (deve ter rows x cols imagens)

        Returns:
            Tupla (resultado, info)
        """
        rows, cols = self.grid_size
        expected_images = rows * cols

        if len(images) != expected_images:
            return None, {
                'status': 'error',
                'message': f"Esperado {expected_images} imagens para grid {rows}x{cols}, recebido {len(images)}"
            }

        try:
            # Passo 1: Fazer stitching horizontal de cada linha
            horizontal_stitches = []

            for row_idx in range(rows):
                start_idx = row_idx * cols
                end_idx = start_idx + cols
                row_images = images[start_idx:end_idx]

                print(f"Stitching linha {row_idx + 1}/{rows}...")
                self.directional_stitcher.direction = 'horizontal'
                row_result, row_info = self.directional_stitcher.stitch(row_images)

                if row_result is None:
                    return None, {
                        'status': 'error',
                        'message': f"Falha no stitching da linha {row_idx + 1}",
                        'row_info': row_info
                    }

                horizontal_stitches.append(row_result)

            # Passo 2: Fazer stitching vertical das linhas
            print("Stitching vertical das linhas...")
            self.directional_stitcher.direction = 'vertical'
            final_result, final_info = self.directional_stitcher.stitch(horizontal_stitches)

            if final_result is None:
                return None, {
                    'status': 'error',
                    'message': 'Falha no stitching vertical final',
                    'final_info': final_info
                }

            return final_result, {
                'status': 'success',
                'grid_size': self.grid_size,
                'images_used': len(images)
            }

        except Exception as e:
            return None, {
                'status': 'error',
                'message': str(e)
            }

    def stitch_2x2(self, images: List[np.ndarray]) -> Tuple[Optional[np.ndarray], dict]:
        """Atalho para grid 2x2"""
        self.grid_size = (2, 2)
        return self.stitch_grid(images)

    def stitch_3x3(self, images: List[np.ndarray]) -> Tuple[Optional[np.ndarray], dict]:
        """Atalho para grid 3x3"""
        self.grid_size = (3, 3)
        return self.stitch_grid(images)

    def stitch_custom_grid(self, images: List[np.ndarray],
                          rows: int, cols: int) -> Tuple[Optional[np.ndarray], dict]:
        """Stitching com grid customizado"""
        self.grid_size = (rows, cols)
        return self.stitch_grid(images)

    def stitch_flexible_grid(self, images: List[np.ndarray],
                            auto_arrange: bool = True) -> Tuple[Optional[np.ndarray], dict]:
        """
        Stitching com grid flexível (tenta arranjar automaticamente)

        Args:
            images: Lista de imagens
            auto_arrange: Se True, tenta determinar melhor arranjo

        Returns:
            Tupla (resultado, info)
        """
        num_images = len(images)

        if auto_arrange:
            # Determinar melhor grid baseado no número de imagens
            if num_images == 4:
                rows, cols = 2, 2
            elif num_images == 6:
                rows, cols = 2, 3
            elif num_images == 9:
                rows, cols = 3, 3
            elif num_images == 8:
                rows, cols = 2, 4
            else:
                # Tentar aproximar um grid quadrado
                cols = int(np.ceil(np.sqrt(num_images)))
                rows = int(np.ceil(num_images / cols))
        else:
            rows, cols = self.grid_size

        print(f"Arranjando {num_images} imagens em grid {rows}x{cols}")
        self.grid_size = (rows, cols)

        # Se temos menos imagens que slots, preencher com preto
        expected = rows * cols
        if num_images < expected:
            print(f"Preenchendo {expected - num_images} slots vazios com preto")
            # Obter tamanho de uma imagem de referência
            h, w = images[0].shape[:2]
            black_img = np.zeros((h, w, 3), dtype=np.uint8)

            # Adicionar imagens pretas
            images = images + [black_img] * (expected - num_images)

        return self.stitch_grid(images)


def stitch_grid_simple(images: List[np.ndarray],
                      rows: int = 2,
                      cols: int = 2,
                      method: Optional[any] = None) -> Tuple[Optional[np.ndarray], dict]:
    """
    Função auxiliar para stitching em grid

    Args:
        images: Lista de imagens
        rows: Número de linhas
        cols: Número de colunas
        method: Método de stitching (opcional)

    Returns:
        Tupla (resultado, info)
    """
    stitcher = GridStitcher(grid_size=(rows, cols), base_method=method)
    return stitcher.stitch_grid(images)
