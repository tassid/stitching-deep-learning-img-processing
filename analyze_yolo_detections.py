"""
Script para analisar detecções YOLO e gerar CSV com contagem de objetos
"""
import json
import os
import pandas as pd
from pathlib import Path
from collections import Counter

def analyze_yolo_folder(folder_path):
    """Analisa uma pasta de detecções YOLO e retorna estatísticas"""
    results = []

    json_files = list(Path(folder_path).glob("*.json"))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Conta objetos por classe
        if 'detections' in data:
            detections = data['detections']

            # Verifica se é o formato com count_by_class
            if 'count_by_class' in detections and 'total_detections' in detections:
                total_objects = detections['total_detections']
                class_counts = detections['count_by_class']
            else:
                # Formato antigo ou diferente
                total_objects = 0
                class_counts = {}

            results.append({
                'arquivo': json_file.stem.replace('detection_', ''),
                'total_objetos': total_objects,
                'classes': dict(class_counts)
            })

    return results

def create_detection_summary_csv(base_folder):
    """Cria CSVs resumindo detecções para todas as pastas yolo_detection"""
    results_folder = Path(base_folder)

    yolo_folders = [f for f in results_folder.iterdir() if f.is_dir() and 'yolo_detection' in f.name]

    for yolo_folder in yolo_folders:
        print(f"Analisando: {yolo_folder.name}")

        detections = analyze_yolo_folder(yolo_folder)

        if detections:
            # Cria DataFrame com todas as classes encontradas
            all_classes = set()
            for det in detections:
                all_classes.update(det['classes'].keys())

            rows = []
            for det in detections:
                row = {
                    'arquivo': det['arquivo'],
                    'total_objetos': det['total_objetos']
                }
                # Adiciona contagem por classe
                for cls in sorted(all_classes):
                    row[f'classe_{cls}'] = det['classes'].get(cls, 0)
                rows.append(row)

            df = pd.DataFrame(rows)
            output_csv = yolo_folder / 'analise_deteccoes.csv'
            df.to_csv(output_csv, index=False)
            print(f"  CSV criado: {output_csv}")
            print(f"  Total de arquivos analisados: {len(detections)}")
            print(f"  Total de objetos detectados: {df['total_objetos'].sum()}")
            print()

if __name__ == '__main__':
    base_folder = r"C:\Users\TASSI\Documents\GitHub\stitching-deep-learning-img-processing\results"
    create_detection_summary_csv(base_folder)
    print("Análise concluída!")
