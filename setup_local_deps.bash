#!/usr/bin/env bash
# Only ONNX Runtime for yolo_3d_node (PCL/g++ come from apt now).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ORT_DIR="${ROOT}/.deps/onnxruntime-linux-x64-1.17.1"
if [[ -d "${ORT_DIR}" ]]; then
  export ONNXRUNTIME_ROOT="${ORT_DIR}"
  export LD_LIBRARY_PATH="${ORT_DIR}/lib:${LD_LIBRARY_PATH}"
else
  echo "WARN: ONNX Runtime not found at ${ORT_DIR}" >&2
fi