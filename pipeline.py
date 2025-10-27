"""
Pipeline para executar todos os algoritmos de stitching e calcular métricas
Permite comparar diferentes métodos automaticamente
"""

import os
import cv2
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import json
from datetime import datetime

from metricas import MetricasImageStitching


class PipelineImageStitching:
    """
    Pipeline para executar todos os algoritmos e avaliar resultados
    """

    def __init__(self, datasets_dir: str = "datasets", resultados_dir: str = "resultados_pipeline"):
        self.datasets_dir = Path(datasets_dir)
        self.resultados_dir = Path(resultados_dir)
        self.resultados_dir.mkdir(exist_ok=True, parents=True)

        self.calc_metricas = MetricasImageStitching()

        # Lista dos 5 conjuntos principais do paper
        self.conjuntos = [
            'stop_sign_1',
            'graffiti_building_1',
            'cars_1',
            'ski_left',
            'P2_front_of_window'
        ]

        # Descobrir algoritmos disponíveis automaticamente
        self.algoritmos = self._descobrir_algoritmos()

        print(f"✅ Pipeline inicializado")
        print(f"   📁 Datasets: {self.datasets_dir}")
        print(f"   💾 Resultados: {self.resultados_dir}")
        print(f"   🔧 Algoritmos disponíveis: {len(self.algoritmos)}")

    def _descobrir_algoritmos(self) -> Dict[str, callable]:
        """
        Descobre automaticamente os algoritmos disponíveis em modules/
        """
        algoritmos = {}

        # Importar módulos disponíveis
        try:
            # Clássicos
            from modules.classic import opencv_stitcher, orb_homography
            algoritmos['opencv_stitcher'] = self._wrap_opencv_stitcher
            algoritmos['orb_homography'] = self._wrap_orb_homography
        except Exception as e:
            print(f"⚠️  Alguns algoritmos clássicos não disponíveis: {e}")

        try:
            # Deep learning
            from modules.deep import loftr
            algoritmos['loftr'] = self._wrap_loftr
        except Exception as e:
            print(f"⚠️  LoFTR não disponível: {e}")

        # Algoritmos de blending
        blend_methods = [
            '5_blend_wavelet',
            '8_blend_laplacian',
            '10_blend_poisson',
            '11_blend_multifocal'
        ]

        for method in blend_methods:
            try:
                algoritmos[method] = self._wrap_module_method(f"modules.classic.{method}")
            except Exception:
                pass

        if not algoritmos:
            print("❌ Nenhum algoritmo descoberto! Verifique modules/classic e modules/deep")

        return algoritmos

    def _wrap_opencv_stitcher(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        """Wrapper para OpenCV Stitcher"""
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch([input1, input2])
        if status != 0:
            raise RuntimeError(f"OpenCV Stitcher falhou com status {status}")
        return pano

    def _wrap_orb_homography(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        """Wrapper para ORB + Homography"""
        # Implementação básica ORB
        orb = cv2.ORB_create(nfeatures=4000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Detectar features
        gray1 = cv2.cvtColor(input1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(input2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            raise RuntimeError("Não foi possível extrair features")

        # Correspondências
        matches = bf.knnMatch(des2, des1, k=2)
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 8:
            raise RuntimeError("Poucas correspondências encontradas")

        # Homografia
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
        pts2 = np.float32([kp2[m.queryIdx].pt for m in good])
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)

        if H is None:
            raise RuntimeError("Falha ao calcular homografia")

        # Warp
        h1, w1 = input1.shape[:2]
        h2, w2 = input2.shape[:2]
        corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
        corners2_t = cv2.perspectiveTransform(corners2, H)
        corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((corners2_t, corners1))

        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        t = [-xmin, -ymin]
        H_t = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

        result = cv2.warpPerspective(input2, H_t @ H, (xmax - xmin, ymax - ymin))
        result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = input1

        return result

    def _wrap_loftr(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        """Wrapper para LoFTR"""
        from modules.deep.loftr import stitch
        return stitch([input1, input2])

    def _wrap_module_method(self, module_name: str):
        """Wrapper genérico para módulos"""
        def wrapper(input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
            import importlib
            mod = importlib.import_module(module_name)
            if hasattr(mod, 'stitch'):
                return mod.stitch([input1, input2])
            else:
                raise NotImplementedError(f"{module_name} não tem função stitch()")
        return wrapper

    def carregar_conjunto(self, nome_conjunto: str, dataset_base: str = "robust_multiple_registrations") -> Dict[str, Optional[np.ndarray]]:
        """
        Carrega imagens de um conjunto

        Args:
            nome_conjunto: Nome do conjunto (ex: stop_sign_1)
            dataset_base: Nome do dataset base

        Returns:
            {'input1': array, 'input2': array, 'ground_truth': array}
        """
        # Tentar múltiplos caminhos possíveis
        possible_paths = [
            self.datasets_dir / dataset_base / nome_conjunto,
            self.datasets_dir / "Image-Stitching-Dataset" / "Robust_Multiple_Registrations" / nome_conjunto,
            self.datasets_dir / nome_conjunto,
        ]

        caminho = None
        for p in possible_paths:
            if p.exists():
                caminho = p
                break

        if caminho is None:
            raise FileNotFoundError(
                f"Conjunto '{nome_conjunto}' não encontrado em:\n" +
                "\n".join(f"  - {p}" for p in possible_paths)
            )

        print(f"📂 Carregando conjunto: {caminho}")

        # Carregar input1 e input2
        input1_path = caminho / 'input1.jpg'
        input2_path = caminho / 'input2.jpg'

        if not input1_path.exists():
            raise FileNotFoundError(f"input1.jpg não encontrado em {caminho}")
        if not input2_path.exists():
            raise FileNotFoundError(f"input2.jpg não encontrado em {caminho}")

        input1 = cv2.imread(str(input1_path))
        input2 = cv2.imread(str(input2_path))

        if input1 is None or input2 is None:
            raise RuntimeError(f"Falha ao ler imagens em {caminho}")

        # Converter BGR -> RGB
        input1 = cv2.cvtColor(input1, cv2.COLOR_BGR2RGB)
        input2 = cv2.cvtColor(input2, cv2.COLOR_BGR2RGB)

        # Ground truth pode não existir
        gt_path = caminho / 'result.jpg'
        ground_truth = None
        if gt_path.exists():
            ground_truth = cv2.imread(str(gt_path))
            ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
            print(f"   ✅ Ground truth encontrado")
        else:
            print(f"   ⚠️  Ground truth não encontrado (métricas PSNR/SSIM serão None)")

        return {
            'input1': input1,
            'input2': input2,
            'ground_truth': ground_truth,
            'caminho': str(caminho)
        }

    def processar_conjunto(self, nome_conjunto: str) -> List[Dict]:
        """
        Processa um conjunto com todos os algoritmos

        Args:
            nome_conjunto: Nome do conjunto

        Returns:
            Lista de dicionários com resultados
        """
        print(f"\n{'=' * 80}")
        print(f"🔄 PROCESSANDO: {nome_conjunto}")
        print(f"{'=' * 80}")

        # Carregar imagens
        try:
            imagens = self.carregar_conjunto(nome_conjunto)
        except FileNotFoundError as e:
            print(f"❌ Erro: {e}")
            print(f"   Pulando conjunto '{nome_conjunto}'...")
            return []

        input1 = imagens['input1']
        input2 = imagens['input2']
        ground_truth = imagens['ground_truth']

        resultados_conjunto = []

        # Executar cada algoritmo
        for nome_algoritmo, funcao_algoritmo in self.algoritmos.items():
            print(f"\n🔧 Executando: {nome_algoritmo}...", end=" ")

            try:
                # Medir tempo
                inicio = time.time()

                # Executar (input em RGB)
                resultado = funcao_algoritmo(input1, input2)

                tempo_total = time.time() - inicio

                print(f"✅ ({tempo_total:.2f}s)")

                # Garantir que resultado está em RGB
                if resultado.shape[2] == 3:
                    # Salvar resultado
                    nome_arquivo = f"{nome_conjunto}_{nome_algoritmo}.jpg"
                    caminho_saida = self.resultados_dir / nome_arquivo

                    # Converter RGB -> BGR para salvar
                    resultado_bgr = cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(caminho_saida), resultado_bgr)

                    # Calcular métricas
                    metricas = self.calc_metricas.calcular_todas(
                        resultado,
                        ground_truth,
                        tempo_total
                    )

                    # Compilar resultado
                    resultado_dict = {
                        'conjunto': nome_conjunto,
                        'algoritmo': nome_algoritmo,
                        'tempo_segundos': metricas['tempo_segundos'],
                        'psnr_db': metricas['psnr_db'],
                        'ssim': metricas['ssim'],
                        'mse': metricas['mse'],
                        'entropia': metricas['entropia'],
                        'nitidez': metricas['nitidez'],
                        'arquivo_saida': nome_arquivo,
                        'largura': resultado.shape[1],
                        'altura': resultado.shape[0]
                    }

                    # Imprimir métricas
                    if resultado_dict['psnr_db'] is not None:
                        print(f"   📊 PSNR: {resultado_dict['psnr_db']:.2f} dB | SSIM: {resultado_dict['ssim']:.4f}")
                    print(f"   📐 Tamanho: {resultado_dict['largura']}x{resultado_dict['altura']} | Entropia: {resultado_dict['entropia']:.4f}")

                    resultados_conjunto.append(resultado_dict)
                else:
                    raise RuntimeError(f"Resultado com formato inesperado: {resultado.shape}")

            except Exception as e:
                print(f"❌ ERRO: {str(e)[:100]}")
                resultado_dict = {
                    'conjunto': nome_conjunto,
                    'algoritmo': nome_algoritmo,
                    'tempo_segundos': None,
                    'psnr_db': None,
                    'ssim': None,
                    'mse': None,
                    'entropia': None,
                    'nitidez': None,
                    'arquivo_saida': None,
                    'largura': None,
                    'altura': None,
                    'erro': str(e)[:200]
                }
                resultados_conjunto.append(resultado_dict)

        return resultados_conjunto

    def executar_pipeline_completo(self, conjuntos: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Executa pipeline completo em todos (ou alguns) conjuntos

        Args:
            conjuntos: Lista de conjuntos (None = todos)

        Returns:
            DataFrame com resultados
        """
        print("\n" + "=" * 80)
        print("🚀 INICIANDO PIPELINE DE AVALIAÇÃO COMPLETA")
        print("=" * 80)

        if conjuntos is None:
            conjuntos = self.conjuntos

        print(f"📋 Conjuntos a processar: {len(conjuntos)}")
        print(f"🔧 Algoritmos a testar: {len(self.algoritmos)}")
        print(f"🧪 Total de experimentos: {len(conjuntos) * len(self.algoritmos)}")

        todos_resultados = []

        # Processar cada conjunto
        for idx, conjunto in enumerate(conjuntos, 1):
            print(f"\n[{idx}/{len(conjuntos)}] Conjunto: {conjunto}")
            resultados = self.processar_conjunto(conjunto)
            todos_resultados.extend(resultados)

        # Salvar resultados em CSV
        df = pd.DataFrame(todos_resultados)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.resultados_dir / f'resultados_metricas_{timestamp}.csv'

        df.to_csv(csv_path, index=False, float_format='%.4f')

        # Salvar também JSON para backup
        json_path = self.resultados_dir / f'resultados_metricas_{timestamp}.json'
        df.to_json(json_path, orient='records', indent=2)

        print(f"\n{'=' * 80}")
        print(f"✅ PIPELINE CONCLUÍDO!")
        print(f"{'=' * 80}")
        print(f"📊 Total de experimentos: {len(todos_resultados)}")
        print(f"💾 Resultados salvos em:")
        print(f"   - CSV: {csv_path}")
        print(f"   - JSON: {json_path}")
        print(f"   - Imagens: {self.resultados_dir}/*.jpg")

        # Imprimir resumo
        self.imprimir_resumo(df)

        return df

    def imprimir_resumo(self, df: pd.DataFrame):
        """
        Imprime resumo dos resultados
        """
        print(f"\n{'=' * 80}")
        print("📈 RESUMO DAS MÉTRICAS (MÉDIAS POR ALGORITMO)")
        print(f"{'=' * 80}")

        # Filtrar apenas sucessos
        df_success = df[df['psnr_db'].notna()]

        if len(df_success) > 0:
            # Agrupar por algoritmo
            resumo = df_success.groupby('algoritmo')[['psnr_db', 'ssim', 'tempo_segundos']].agg(['mean', 'std'])
            print("\n", resumo.to_string())

            print(f"\n{'=' * 80}")
            print("🏆 MELHORES RESULTADOS")
            print(f"{'=' * 80}")

            # Melhor PSNR
            if 'psnr_db' in df_success.columns:
                idx_max_psnr = df_success['psnr_db'].idxmax()
                melhor_psnr = df_success.loc[idx_max_psnr]
                print(f"\n🥇 Melhor PSNR: {melhor_psnr['psnr_db']:.2f} dB")
                print(f"   Algoritmo: {melhor_psnr['algoritmo']}")
                print(f"   Conjunto: {melhor_psnr['conjunto']}")

            # Melhor SSIM
            if 'ssim' in df_success.columns:
                idx_max_ssim = df_success['ssim'].idxmax()
                melhor_ssim = df_success.loc[idx_max_ssim]
                print(f"\n🥇 Melhor SSIM: {melhor_ssim['ssim']:.4f}")
                print(f"   Algoritmo: {melhor_ssim['algoritmo']}")
                print(f"   Conjunto: {melhor_ssim['conjunto']}")

            # Mais rápido
            if 'tempo_segundos' in df_success.columns:
                idx_min_tempo = df_success['tempo_segundos'].idxmin()
                mais_rapido = df_success.loc[idx_min_tempo]
                print(f"\n⚡ Mais Rápido: {mais_rapido['tempo_segundos']:.2f}s")
                print(f"   Algoritmo: {mais_rapido['algoritmo']}")
                print(f"   Conjunto: {mais_rapido['conjunto']}")

        # Estatísticas de erros
        df_errors = df[df['psnr_db'].isna()]
        if len(df_errors) > 0:
            print(f"\n{'=' * 80}")
            print(f"⚠️  EXPERIMENTOS COM ERROS: {len(df_errors)}")
            print(f"{'=' * 80}")
            for _, row in df_errors.iterrows():
                erro_msg = row.get('erro', 'Erro desconhecido')[:80]
                print(f"  ❌ {row['algoritmo']} em {row['conjunto']}: {erro_msg}")


# ===== CLI =====
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline de comparação de algoritmos de image stitching"
    )
    parser.add_argument(
        "--datasets",
        default="datasets",
        help="Diretório com os datasets (padrão: datasets)"
    )
    parser.add_argument(
        "--output",
        default="resultados_pipeline",
        help="Diretório de saída (padrão: resultados_pipeline)"
    )
    parser.add_argument(
        "--conjuntos",
        nargs="+",
        help="Lista de conjuntos específicos a processar (padrão: todos)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Modo teste: processar apenas 1 conjunto"
    )

    args = parser.parse_args()

    # Criar pipeline
    pipeline = PipelineImageStitching(
        datasets_dir=args.datasets,
        resultados_dir=args.output
    )

    # Determinar conjuntos
    if args.test:
        # Modo teste: apenas primeiro conjunto
        conjuntos = [pipeline.conjuntos[0]]
        print("🧪 MODO TESTE: Processando apenas 1 conjunto")
    elif args.conjuntos:
        conjuntos = args.conjuntos
    else:
        conjuntos = None  # Todos

    # Executar
    resultados_df = pipeline.executar_pipeline_completo(conjuntos)

    print("\n✅ Pipeline finalizado com sucesso!")
    print(f"📊 Use visualizar_resultados.py para análise detalhada")
