"""
YOLO Object Detector
Detecta objetos (carros, pessoas, etc) em imagens e panoramas
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json


class YOLODetector:
    """
    Detector de objetos usando YOLO (v5, v8, ou v11)
    """

    def __init__(self, model_version: str = 'yolov8', model_size: str = 'n', device: str = 'cuda'):
        """
        Inicializa o detector YOLO

        Args:
            model_version: Versão do YOLO ('yolov5', 'yolov8', 'yolov11')
            model_size: Tamanho do modelo ('n', 's', 'm', 'l', 'x')
            device: Dispositivo ('cuda' ou 'cpu')
        """
        self.model_version = model_version
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Carrega o modelo YOLO"""
        try:
            if self.model_version == 'yolov5':
                import torch
                self.model = torch.hub.load('ultralytics/yolov5', f'yolov5{self.model_size}')
                self.model.to(self.device)
            elif self.model_version in ['yolov8', 'yolov11']:
                from ultralytics import YOLO
                # YOLO 11 usa nomenclatura sem 'v': yolo11 ao invés de yolov11
                model_name = self.model_version.replace('yolov', 'yolo')
                self.model = YOLO(f'{model_name}{self.model_size}.pt')
            else:
                raise ValueError(f"Versão não suportada: {self.model_version}")

            print(f"✓ Modelo {self.model_version}{self.model_size} carregado com sucesso")
        except Exception as e:
            print(f"✗ Erro ao carregar modelo: {e}")
            raise

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25,
               classes: Optional[List[int]] = None) -> Dict:
        """
        Detecta objetos em uma imagem

        Args:
            image: Imagem (BGR ou RGB)
            conf_threshold: Limiar de confiança
            classes: Lista de classes para detectar (None = todas)
                    Ex: [0] = pessoa, [2] = carro

        Returns:
            Dict com resultados da detecção
        """
        if self.model is None:
            raise RuntimeError("Modelo não carregado")

        # Realizar detecção
        if self.model_version == 'yolov5':
            results = self.model(image)
            detections = results.pandas().xyxy[0]
        else:  # yolov8 ou yolov11
            results = self.model(image, conf=conf_threshold, classes=classes, verbose=False)
            detections = results[0].boxes

        # Processar resultados
        return self._process_results(results, detections)

    def _process_results(self, results, detections) -> Dict:
        """Processa resultados da detecção"""
        output = {
            'boxes': [],
            'confidences': [],
            'classes': [],
            'class_names': [],
            'count_by_class': {}
        }

        if self.model_version == 'yolov5':
            for _, row in detections.iterrows():
                output['boxes'].append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                output['confidences'].append(row['confidence'])
                output['classes'].append(int(row['class']))
                output['class_names'].append(row['name'])

                # Contar por classe
                class_name = row['name']
                output['count_by_class'][class_name] = output['count_by_class'].get(class_name, 0) + 1
        else:
            for box in detections:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = results[0].names[cls]

                output['boxes'].append([float(x1), float(y1), float(x2), float(y2)])
                output['confidences'].append(conf)
                output['classes'].append(cls)
                output['class_names'].append(class_name)

                # Contar por classe
                output['count_by_class'][class_name] = output['count_by_class'].get(class_name, 0) + 1

        output['total_detections'] = len(output['boxes'])
        return output

    def detect_cars(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict:
        """Detecta apenas carros"""
        # Classe 2 = car no COCO dataset
        return self.detect(image, conf_threshold, classes=[2])

    def detect_persons(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict:
        """Detecta apenas pessoas"""
        # Classe 0 = person no COCO dataset
        return self.detect(image, conf_threshold, classes=[0])

    def detect_cars_and_persons(self, image: np.ndarray, conf_threshold: float = 0.25) -> Dict:
        """Detecta carros e pessoas"""
        # Classes 0 = person, 2 = car
        return self.detect(image, conf_threshold, classes=[0, 2])

    def visualize(self, image: np.ndarray, detections: Dict,
                  draw_labels: bool = True) -> np.ndarray:
        """
        Desenha as detecções na imagem

        Args:
            image: Imagem original
            detections: Resultado do método detect()
            draw_labels: Se deve desenhar labels

        Returns:
            Imagem com detecções desenhadas
        """
        result_img = image.copy()

        # Cores por classe (BGR)
        colors = {
            'person': (0, 255, 0),    # Verde
            'car': (255, 0, 0),       # Azul
            'default': (0, 0, 255)    # Vermelho
        }

        for i, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            conf = detections['confidences'][i]
            class_name = detections['class_names'][i]

            # Escolher cor
            color = colors.get(class_name, colors['default'])

            # Desenhar caixa
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)

            # Desenhar label
            if draw_labels:
                label = f"{class_name} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(result_img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
                cv2.putText(result_img, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Adicionar contagem total
        total_text = f"Total: {detections['total_detections']}"
        cv2.putText(result_img, total_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return result_img

    def compare_detections(self, individual_images: List[np.ndarray],
                          panorama: np.ndarray,
                          conf_threshold: float = 0.25,
                          classes: Optional[List[int]] = None) -> Dict:
        """
        Compara detecções entre imagens individuais e panorama

        Args:
            individual_images: Lista de imagens individuais
            panorama: Imagem panorâmica gerada
            conf_threshold: Limiar de confiança
            classes: Classes para detectar

        Returns:
            Dict com comparação de resultados
        """
        # Detectar em imagens individuais
        individual_detections = []
        individual_total = 0

        for img in individual_images:
            det = self.detect(img, conf_threshold, classes)
            individual_detections.append(det)
            individual_total += det['total_detections']

        # Detectar no panorama
        panorama_detection = self.detect(panorama, conf_threshold, classes)

        # Compilar resultados
        comparison = {
            'individual_images': {
                'detections': individual_detections,
                'total_detections': individual_total,
                'count_by_class': {}
            },
            'panorama': {
                'detections': panorama_detection,
                'total_detections': panorama_detection['total_detections'],
                'count_by_class': panorama_detection['count_by_class']
            },
            'analysis': {
                'duplicates_likely': individual_total > panorama_detection['total_detections'],
                'improvement': panorama_detection['total_detections'] - individual_total,
                'improvement_percentage': 0
            }
        }

        # Calcular melhoria
        if individual_total > 0:
            improvement_pct = ((panorama_detection['total_detections'] - individual_total) /
                             individual_total) * 100
            comparison['analysis']['improvement_percentage'] = improvement_pct

        return comparison

    def save_results(self, output_path: str, detections: Dict,
                    image_path: Optional[str] = None):
        """
        Salva resultados da detecção em arquivo JSON

        Args:
            output_path: Caminho do arquivo de saída
            detections: Resultado do método detect()
            image_path: Caminho da imagem original (opcional)
        """
        output = {
            'image_path': image_path,
            'model': f"{self.model_version}{self.model_size}",
            'detections': detections
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Resultados salvos em: {output_path}")


# Classes COCO dataset (referência)
COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    # ... (adicione mais conforme necessário)
}
