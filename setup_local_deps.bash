#!/usr/bin/env bash
# RKNN runtime for yolo_3d_node (PCL/g++ come from apt).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RKNN_DIR="${ROOT}/.deps/rknpu2"
if [[ -f "${RKNN_DIR}/lib/librknnrt.so" ]]; then
  export RKNN_ROOT="${RKNN_DIR}"
  export LD_LIBRARY_PATH="${RKNN_DIR}/lib:${LD_LIBRARY_PATH:-}"
elif [[ -f /usr/lib/librknnrt.so ]]; then
  export LD_LIBRARY_PATH="/usr/lib:${LD_LIBRARY_PATH:-}"
else
  echo "WARN: librknnrt.so not found under ${RKNN_DIR}/lib or /usr/lib" >&2
fi
