"""
Script para criar colagens comparativas das detecções YOLO
"""
import cv2
import numpy as np
from pathlib import Path
import json

def add_text_to_image(img, text, position='top'):
    """Adiciona texto à imagem"""
    h, w = img.shape[:2]

    # Cria barra para o texto
    bar_height = 60
    bar = np.ones((bar_height, w, 3), dtype=np.uint8) * 40  # Cinza escuro

    # Adiciona texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Calcula posição para centralizar
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2

    cv2.putText(bar, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    # Combina barra com imagem
    if position == 'top':
        return np.vstack([bar, img])
    else:
        return np.vstack([img, bar])

def create_method_collage(detection_folder, output_folder, dataset_name='biker_mural_1'):
    """Cria colagem comparando diferentes métodos para um dataset"""
    detection_path = Path(detection_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Procura imagens de detecção para o dataset especificado
    detection_images = list(detection_path.glob(f"detection_{dataset_name}_*.jpg"))

    if not detection_images:
        print(f"Nenhuma detecção encontrada para {dataset_name}")
        return

    # Agrupa por método
    methods_data = {}
    for img_path in detection_images:
        # Extrai nome do método do arquivo
        parts = img_path.stem.replace(f'detection_{dataset_name}_', '').split('_')
        if len(parts) >= 2:
            method_name = '_'.join(parts)

            # Lê imagem
            img = cv2.imread(str(img_path))
            if img is not None:
                # Lê JSON para obter contagem
                json_path = img_path.with_suffix('.json')
                total_objs = 0
                if json_path.exists():
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        if 'detections' in data and 'total_detections' in data['detections']:
                            total_objs = data['detections']['total_detections']

                methods_data[method_name] = {
                    'image': img,
                    'total': total_objs
                }

    if not methods_data:
        print(f"Nenhum método encontrado para {dataset_name}")
        return

    # Ordena métodos por total de objetos detectados (decrescente)
    sorted_methods = sorted(methods_data.items(), key=lambda x: x[1]['total'], reverse=True)

    # Seleciona top 6 métodos para a colagem
    num_methods = min(6, len(sorted_methods))
    selected = sorted_methods[:num_methods]

    # Redimensiona imagens para tamanho padrão
    target_width = 800
    resized_images = []

    for method_name, data in selected:
        img = data['image']
        total = data['total']

        # Redimensiona mantendo aspect ratio
        h, w = img.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (target_width, new_h))

        # Adiciona label com nome do método e contagem
        label = f"{method_name.replace('_', ' ').title()} - {total} objetos"
        img_labeled = add_text_to_image(img_resized, label)

        resized_images.append(img_labeled)

    # Cria colagem em grid 2x3
    if num_methods <= 3:
        # Uma linha
        collage = np.hstack(resized_images)
    else:
        # Duas linhas
        mid = (num_methods + 1) // 2
        row1 = resized_images[:mid]
        row2 = resized_images[mid:]

        # Equaliza larguras das linhas
        while len(row2) < len(row1):
            # Adiciona imagem branca
            h = row2[0].shape[0] if row2 else row1[0].shape[0]
            row2.append(np.ones((h, target_width, 3), dtype=np.uint8) * 255)

        collage_row1 = np.hstack(row1)
        collage_row2 = np.hstack(row2)

        # Equaliza alturas
        h1, h2 = collage_row1.shape[0], collage_row2.shape[0]
        if h1 != h2:
            max_h = max(h1, h2)
            if h1 < max_h:
                pad = np.ones((max_h - h1, collage_row1.shape[1], 3), dtype=np.uint8) * 255
                collage_row1 = np.vstack([collage_row1, pad])
            if h2 < max_h:
                pad = np.ones((max_h - h2, collage_row2.shape[1], 3), dtype=np.uint8) * 255
                collage_row2 = np.vstack([collage_row2, pad])

        collage = np.vstack([collage_row1, collage_row2])

    # Salva colagem
    output_file = output_path / f"collage_methods_{dataset_name}.jpg"
    cv2.imwrite(str(output_file), collage)
    print(f"Colagem criada: {output_file}")
    print(f"  {num_methods} métodos comparados")
    print(f"  Dimensões: {collage.shape[1]}x{collage.shape[0]}")

def create_dataset_collage(detection_folder, output_folder, method_name='rank01'):
    """Cria colagem comparando diferentes datasets para um método"""
    detection_path = Path(detection_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Procura imagens de detecção para o método especificado
    detection_images = list(detection_path.glob(f"detection_{method_name}_*.jpg"))

    if not detection_images:
        print(f"Nenhuma detecção encontrada para método {method_name}")
        return

    # Processa cada dataset
    datasets_data = {}
    for img_path in detection_images:
        # Extrai nome do dataset
        dataset = img_path.stem.replace(f'detection_{method_name}_', '').rsplit('_', 2)[0]

        # Lê imagem
        img = cv2.imread(str(img_path))
        if img is not None:
            # Lê JSON
            json_path = img_path.with_suffix('.json')
            total_objs = 0
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    if 'detections' in data and 'total_detections' in data['detections']:
                        total_objs = data['detections']['total_detections']

            datasets_data[dataset] = {
                'image': img,
                'total': total_objs
            }

    if not datasets_data:
        print(f"Nenhum dataset encontrado para método {method_name}")
        return

    # Ordena datasets por total de objetos (decrescente)
    sorted_datasets = sorted(datasets_data.items(), key=lambda x: x[1]['total'], reverse=True)

    # Seleciona top 6 datasets
    num_datasets = min(6, len(sorted_datasets))
    selected = sorted_datasets[:num_datasets]

    # Redimensiona imagens
    target_width = 600
    resized_images = []

    for dataset, data in selected:
        img = data['image']
        total = data['total']

        h, w = img.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (target_width, new_h))

        label = f"{dataset.replace('_', ' ').title()} - {total} objetos"
        img_labeled = add_text_to_image(img_resized, label)

        resized_images.append(img_labeled)

    # Cria colagem em grid 2x3
    if num_datasets <= 3:
        # Equaliza alturas antes de empilhar
        max_h = max(img.shape[0] for img in resized_images)
        equalized = []
        for img in resized_images:
            if img.shape[0] < max_h:
                pad = np.ones((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
                img = np.vstack([img, pad])
            equalized.append(img)
        collage = np.hstack(equalized)
    else:
        mid = (num_datasets + 1) // 2
        row1 = resized_images[:mid]
        row2 = resized_images[mid:]

        while len(row2) < len(row1):
            h = row2[0].shape[0] if row2 else row1[0].shape[0]
            row2.append(np.ones((h, target_width, 3), dtype=np.uint8) * 255)

        # Equaliza alturas dentro de cada linha
        max_h1 = max(img.shape[0] for img in row1)
        row1_eq = []
        for img in row1:
            if img.shape[0] < max_h1:
                pad = np.ones((max_h1 - img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
                img = np.vstack([img, pad])
            row1_eq.append(img)

        max_h2 = max(img.shape[0] for img in row2)
        row2_eq = []
        for img in row2:
            if img.shape[0] < max_h2:
                pad = np.ones((max_h2 - img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
                img = np.vstack([img, pad])
            row2_eq.append(img)

        collage_row1 = np.hstack(row1_eq)
        collage_row2 = np.hstack(row2_eq)

        h1, h2 = collage_row1.shape[0], collage_row2.shape[0]
        if h1 != h2:
            max_h = max(h1, h2)
            if h1 < max_h:
                pad = np.ones((max_h - h1, collage_row1.shape[1], 3), dtype=np.uint8) * 255
                collage_row1 = np.vstack([collage_row1, pad])
            if h2 < max_h:
                pad = np.ones((max_h - h2, collage_row2.shape[1], 3), dtype=np.uint8) * 255
                collage_row2 = np.vstack([collage_row2, pad])

        collage = np.vstack([collage_row1, collage_row2])

    output_file = output_path / f"collage_datasets_{method_name}.jpg"
    cv2.imwrite(str(output_file), collage)
    print(f"Colagem criada: {output_file}")
    print(f"  {num_datasets} datasets comparados")
    print(f"  Dimensões: {collage.shape[1]}x{collage.shape[0]}")

if __name__ == '__main__':
    base_path = r"C:\Users\TASSI\Documents\GitHub\stitching-deep-learning-img-processing\results"

    # Cria colagens para pasta yolo_detection_20251103_201954 (diferentes métodos, mesmo dataset)
    detection_folder1 = Path(base_path) / "yolo_detection_20251103_201954"
    if detection_folder1.exists():
        print("\n=== Criando colagem de métodos (biker_mural_1) ===")
        create_method_collage(detection_folder1, detection_folder1 / "collages", "biker_mural_1")

    # Cria colagens para pasta yolo_detection_20251103_233827 (diferentes datasets)
    detection_folder2 = Path(base_path) / "yolo_detection_20251103_233827"
    if detection_folder2.exists():
        print("\n=== Criando colagens de datasets ===")
        create_dataset_collage(detection_folder2, detection_folder2 / "collages", "rank01")
        create_dataset_collage(detection_folder2, detection_folder2 / "collages", "rank02")
        create_dataset_collage(detection_folder2, detection_folder2 / "collages", "rank03")

    print("\n=== Colagens criadas com sucesso! ===")
