#!/usr/bin/env bash
# Export yolov8n.pt -> ONNX for yolo_3d_node (ONNX Runtime).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/src/rgbd_detection/models/yolov8n.onnx"
PT="${1:-/home/yuhui/code/RGBD_3D_PRO/yolov8n.pt}"

python3 - <<PY
from ultralytics import YOLO
model = YOLO("${PT}")
# opset 12 is widely compatible; ORT C++ node does not need simplify.
path = model.export(format="onnx", imgsz=640, simplify=False, opset=12)
print(path)
PY

# ultralytics writes next to the .pt by default
if [[ -f "${PT%.*}.onnx" ]]; then
  cp -v "${PT%.*}.onnx" "${OUT}"
fi
ls -lh "${OUT}"
