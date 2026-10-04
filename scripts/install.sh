#!/usr/bin/env bash
# Instalasi dependensi untuk Linux / Colab / Kaggle (Windows: lihat docs/INSTALL.md).
#   bash scripts/install.sh [auto|cpu|gpu]
# Urutan penting: DocAligner & capybara dipasang --no-deps karena metadata capybara 0.6.0
# meminta onnxruntime_gpu==1.20.1 yang tidak ada di PyPI.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-auto}"
# TORCH_GPU: pasang torch CUDA bila torch belum ada. ONNX: runtime untuk DocAligner.
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

# 2) dependensi inti + ONNX Runtime + dependensi capybara
$PIP install -r requirements/base.txt -r "requirements/onnx-$ONNX.txt" -r requirements/docaligner-deps.txt

# 3) library sistem libturbojpeg (dibutuhkan capybara saat import)
if [ "$(uname -s)" = "Linux" ] && ! python -c "from turbojpeg import TurboJPEG; TurboJPEG()" 2>/dev/null; then
  echo ">> memasang libturbojpeg"
  SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
  $SUDO apt-get update -qq || true
  $SUDO apt-get install -y -qq libturbojpeg0 2>/dev/null || $SUDO apt-get install -y -qq libturbojpeg
fi

# 3b) pyheif (hanya Linux): capybara 0.6.0 meng-import-nya tanpa pengaman. Python 3.13 tidak punya wheel,
#     jadi dibangun dari source (butuh libheif-dev). Bila tetap gagal, pasang stub agar import berjalan
#     (hanya membaca file .heic/.heif yang tidak akan berfungsi).
if [ "$(uname -s)" = "Linux" ] && ! python -c "import pyheif" 2>/dev/null; then
  $PIP install pyheif || {
    echo ">> pyheif gagal dipasang, mencoba libheif-dev"
    SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
    ($SUDO apt-get install -y -qq libheif-dev build-essential libffi-dev && $PIP install pyheif) || {
      echo ">> PERINGATAN: pyheif tidak tersedia; memasang stub (file .heic/.heif tidak didukung)"
      python - <<'PY'
import os, site
p = os.path.join(site.getsitepackages()[0], "pyheif.py")
open(p, "w").write("# stub dari scripts/install.sh\ndef read(*a, **k):\n    raise ImportError('pyheif tidak terpasang: .heic/.heif tidak didukung')\n")
print("stub ditulis:", p)
PY
    }
  }
fi

# 4) DocAligner + capybara tanpa dependensi (sudah dipasang manual di atas)
$PIP install --no-deps -r requirements/docaligner.txt

# 5) verifikasi
python - <<'PY'
import numpy, torch, cv2
from capybara import Backend
from docaligner import DocAligner, ModelType
import onnxruntime as ort
print("numpy", numpy.__version__, "| torch", torch.__version__, "| cv2", cv2.__version__)
print("onnxruntime providers:", ort.get_available_providers())
print("OK: DocAligner dapat diimpor")
PY
echo ">> Selesai. Di Colab/Kaggle: restart runtime/kernel bila ada paket (mis. numpy) yang diganti, sebelum menjalankan kode."
