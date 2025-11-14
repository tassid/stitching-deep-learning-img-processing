import json
from pathlib import Path

# Caminho da pasta com os JSONs
folder = Path(r"C:\Users\TASSI\Documents\GitHub\stitching-deep-learning-img-processing\results\yolo_detection_20251112_185921")
out_file = folder / "report.txt"

with out_file.open("w", encoding="utf-8") as f:
    for jf in sorted(folder.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            pretty = json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            pretty = f"<<ERRO AO LER JSON: {e}>>"
        f.write(f"{jf.stem}\n")
        f.write(pretty + "\n\n")
        f.write("proximo\n\n")

print(f"Relatório salvo em: {out_file}")
