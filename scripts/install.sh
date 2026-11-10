#!/usr/bin/env bash
# Instalasi dependensi untuk Linux / Colab / Kaggle (Windows: lihat docs/INSTALL.md).
#   bash scripts/install.sh [auto|cpu|gpu]
# DocAligner dijalankan langsung lewat onnxruntime (src/image_alignment/corner_model.py): tidak ada lagi paket
# docaligner/capybara, libturbojpeg, atau pyheif yang harus dipasang.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-auto}"
# TORCH_GPU: pasang torch CUDA bila torch belum ada. ONNX: runtime untuk CornerDetector.
# `auto` memakai ONNX CPU walau ada GPU: model DocAligner kecil, dan onnxruntime-gpu 1.20 butuh CUDA 12
# (libcublas.so.12) yang tidak ada di semua Colab/Kaggle; bila gagal memuat, pesan error CUDA muncul.
# YOLO dan Donut tetap memakai GPU lewat torch. Gunakan `gpu` hanya jika CUDA 12 + cuDNN 9 tersedia.
case "$MODE" in
  auto) if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then TORCH_GPU=1; else TORCH_GPU=0; fi; ONNX=cpu ;;
  cpu)  TORCH_GPU=0; ONNX=cpu ;;
  gpu)  TORCH_GPU=1; ONNX=gpu ;;
  *)    echo "Mode harus auto|cpu|gpu"; exit 1 ;;
esac
echo ">> mode: $MODE (torch GPU=$TORCH_GPU, onnxruntime=$ONNX)"

if command -v uv >/dev/null 2>&1; then PIP="uv pip"; else PIP="python -m pip"; fi

# 1) torch: jangan ditimpa bila sudah ada (Colab/Kaggle)
if python -c "import torch" 2>/dev/null; then
  echo ">> torch sudah terpasang: $(python -c 'import torch; print(torch.__version__)')"
elif [ "$TORCH_GPU" = "0" ]; then
  $PIP install "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cpu
else
  $PIP install "torch==2.6.0" "torchvision==0.21.0"
fi

# 2) dependensi inti + ONNX Runtime
$PIP install -r requirements/base.txt -r "requirements/onnx-$ONNX.txt"

# 3) verifikasi
python - <<'PY'
import numpy, torch, cv2
import onnxruntime as ort
from src.image_alignment import CornerDetector, warp_image
print("numpy", numpy.__version__, "| torch", torch.__version__, "| cv2", cv2.__version__)
print("onnxruntime providers:", ort.get_available_providers())
print("OK: modul image_alignment dapat diimpor")
PY
echo ">> Selesai. Di Colab/Kaggle: restart runtime/kernel bila ada paket (mis. numpy) yang diganti, sebelum menjalankan kode."
