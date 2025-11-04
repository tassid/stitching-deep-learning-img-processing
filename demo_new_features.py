"""
Demo dos Novos Recursos
Demonstra uso dos módulos: detecção YOLO, stitching direcional/grid, vídeo

Este script NÃO afeta o código existente - é totalmente modular
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List

# Importar novos módulos
from modules.detection import YOLODetector
from modules.advanced_stitching import GridStitcher, DirectionalStitcher
from modules.video import VideoProcessor, extract_frames


def demo_yolo_detection():
    """Demonstra detecção com YOLO"""
    print("\n" + "="*60)
    print("DEMO 1: Detecção de Objetos com YOLO")
    print("="*60)

    # Carregar detector
    detector = YOLODetector(model_version='yolov8', model_size='n', device='cuda')

    # Exemplo com imagem
    # image_path = "path/to/image.jpg"
    # image = cv2.imread(image_path)

    # Simulando com imagem de exemplo
    print("\n1. Detectar carros:")
    # detections_cars = detector.detect_cars(image)
    # print(f"  Carros detectados: {detections_cars['count_by_class']}")

    print("\n2. Detectar pessoas:")
    # detections_persons = detector.detect_persons(image)
    # print(f"  Pessoas detectadas: {detections_persons['count_by_class']}")

    print("\n3. Detectar carros E pessoas:")
    # detections_both = detector.detect_cars_and_persons(image)
    # print(f"  Total detectado: {detections_both['total_detections']}")
    # print(f"  Por classe: {detections_both['count_by_class']}")

    # Visualizar
    # result_img = detector.visualize(image, detections_both)
    # cv2.imwrite("results/detection_demo.jpg", result_img)

    print("\n✓ Demo YOLO concluída")


def demo_directional_stitching():
    """Demonstra stitching direcional (horizontal/vertical)"""
    print("\n" + "="*60)
    print("DEMO 2: Stitching Direcional")
    print("="*60)

    # Stitcher direcional
    stitcher_h = DirectionalStitcher(direction='horizontal')
    stitcher_v = DirectionalStitcher(direction='vertical')

    # Exemplo com imagens
    # images = [cv2.imread(f"dataset/img{i}.jpg") for i in range(3)]

    print("\n1. Stitching Horizontal:")
    # pano_h, info_h = stitcher_h.stitch(images)
    # if pano_h is not None:
    #     cv2.imwrite("results/panorama_horizontal.jpg", pano_h)
    #     print(f"  ✓ Panorama criado: {pano_h.shape}")

    print("\n2. Stitching Vertical:")
    # pano_v, info_v = stitcher_v.stitch(images)
    # if pano_v is not None:
    #     cv2.imwrite("results/panorama_vertical.jpg", pano_v)
    #     print(f"  ✓ Panorama criado: {pano_v.shape}")

    print("\n✓ Demo stitching direcional concluída")


def demo_grid_stitching():
    """Demonstra stitching em grid (2x2, 3x3)"""
    print("\n" + "="*60)
    print("DEMO 3: Stitching em Grid")
    print("="*60)

    # Grid stitcher
    stitcher = GridStitcher()

    print("\n1. Grid 2x2 (4 imagens):")
    # images_2x2 = [cv2.imread(f"dataset/2x2/img{i}.jpg") for i in range(4)]
    # pano_2x2, info_2x2 = stitcher.stitch_2x2(images_2x2)
    # if pano_2x2 is not None:
    #     cv2.imwrite("results/panorama_2x2.jpg", pano_2x2)
    #     print(f"  ✓ Grid 2x2 criado: {pano_2x2.shape}")

    print("\n2. Grid 3x3 (9 imagens):")
    # images_3x3 = [cv2.imread(f"dataset/3x3/img{i}.jpg") for i in range(9)]
    # pano_3x3, info_3x3 = stitcher.stitch_3x3(images_3x3)
    # if pano_3x3 is not None:
    #     cv2.imwrite("results/panorama_3x3.jpg", pano_3x3)
    #     print(f"  ✓ Grid 3x3 criado: {pano_3x3.shape}")

    print("\n3. Grid flexível (auto-arranjo):")
    # images_flex = [cv2.imread(f"dataset/flex/img{i}.jpg") for i in range(6)]
    # pano_flex, info_flex = stitcher.stitch_flexible_grid(images_flex, auto_arrange=True)
    # print(f"  Grid usado: {info_flex.get('grid_size', 'N/A')}")

    print("\n✓ Demo grid stitching concluída")


def demo_video_processing():
    """Demonstra processamento de vídeo"""
    print("\n" + "="*60)
    print("DEMO 4: Processamento de Vídeo")
    print("="*60)

    # Exemplo com vídeo
    # video_path = "path/to/video.mp4"

    print("\n1. Extrair frames a cada 10 frames:")
    # processor = VideoProcessor(video_path)
    # frames = processor.extract_frames_by_interval(interval=10, max_frames=20)
    # print(f"  ✓ Extraídos {len(frames)} frames")

    print("\n2. Extrair keyframes:")
    # keyframes = processor.extract_key_frames(threshold=30.0, max_frames=15)
    # print(f"  ✓ Extraídos {len(keyframes)} keyframes")

    print("\n3. Salvar frames em disco:")
    # saved = processor.extract_and_save_frames(
    #     output_dir="results/video_frames",
    #     interval=15,
    #     prefix="frame"
    # )
    # print(f"  ✓ {len(saved)} frames salvos")

    print("\n✓ Demo vídeo concluída")


def demo_integrated_workflow():
    """Demonstra workflow integrado: vídeo -> stitching -> detecção"""
    print("\n" + "="*60)
    print("DEMO 5: Workflow Integrado")
    print("="*60)

    print("\nFluxo: Vídeo → Extração de Frames → Stitching → Detecção")

    # 1. Extrair frames de vídeo
    print("\n1. Extraindo frames do vídeo...")
    # processor = VideoProcessor("path/to/video.mp4")
    # frames = processor.extract_frames_by_interval(interval=10, max_frames=4)

    # 2. Fazer stitching em grid
    print("\n2. Criando panorama 2x2...")
    # stitcher = GridStitcher(grid_size=(2, 2))
    # panorama, info = stitcher.stitch_grid(frames)

    # 3. Detectar objetos no panorama
    print("\n3. Detectando objetos no panorama...")
    # detector = YOLODetector(model_version='yolov8', model_size='n')
    # detections = detector.detect_cars_and_persons(panorama)

    # 4. Visualizar resultado
    # result = detector.visualize(panorama, detections)
    # cv2.imwrite("results/workflow_final.jpg", result)

    print("\n✓ Workflow integrado concluído")


def demo_with_datasets():
    """Demonstra uso com novos datasets"""
    print("\n" + "="*60)
    print("DEMO 6: Novos Datasets")
    print("="*60)

    datasets = {
        'APAP_case17': r"C:\datasets\APAP Dataset (2013)\images2\case17",
        'APAP_case26': r"C:\datasets\APAP Dataset (2013)\images2\case26",
        'DeepH': r"C:\datasets\Deep Learning Image Stitching\Content-Aware-DeepH-Data (2020)\Contant-Aware-DeepH-Data\Data\Test"
    }

    for name, path in datasets.items():
        print(f"\n{name}:")
        dataset_path = Path(path)

        if dataset_path.exists():
            # Listar imagens
            images = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
            print(f"  ✓ Encontradas {len(images)} imagens")

            # Carregar primeiras 4 para teste
            if len(images) >= 4:
                # imgs = [cv2.imread(str(img)) for img in images[:4]]
                # stitcher = GridStitcher(grid_size=(2, 2))
                # pano, info = stitcher.stitch_grid(imgs)
                # if pano is not None:
                #     output = f"results/{name}_panorama.jpg"
                #     cv2.imwrite(output, pano)
                #     print(f"  ✓ Panorama salvo: {output}")
                pass
        else:
            print(f"  ✗ Dataset não encontrado: {path}")

    print("\n✓ Demo datasets concluída")


def main():
    """Executa todas as demos"""
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO DOS NOVOS RECURSOS")
    print("Módulos: YOLO Detection, Directional/Grid Stitching, Video Processing")
    print("="*60)

    # Criar diretório de resultados
    Path("results").mkdir(exist_ok=True)

    try:
        # Executar demos
        demo_yolo_detection()
        demo_directional_stitching()
        demo_grid_stitching()
        demo_video_processing()
        demo_integrated_workflow()
        demo_with_datasets()

        print("\n" + "="*60)
        print("✓ TODAS AS DEMOS CONCLUÍDAS COM SUCESSO!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Erro durante demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("\n[MODO DEMO - Remova os comentários para executar com dados reais]")
    print("Este script demonstra como usar os novos módulos.")
    print("Descomente as linhas de código para executar com suas imagens/vídeos.\n")

    # Para executar:
    # 1. Descomente as linhas que carregam imagens/vídeos
    # 2. Ajuste os caminhos para seus arquivos
    # 3. Execute: python demo_new_features.py

    main()
