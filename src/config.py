"""Pengaturan lintas-modul: pemilihan device dilakukan di satu tempat ini.

Prioritas: argumen `device` > variabel lingkungan `IDCARD_DEVICE` > otomatis (CUDA bila tersedia, selain itu CPU).
Nilai yang dikenal: "auto", "cpu", "cuda", "cuda:N". Meminta CUDA saat tidak tersedia tidak menggagalkan proses:
dipakai CPU dengan peringatan, supaya skrip yang sama jalan di Colab/Kaggle dengan atau tanpa GPU.
"""
import os
import warnings

ENV_DEVICE = "IDCARD_DEVICE"


def get_device(device=None):
    """Kembalikan nama device torch ("cpu", "cuda", atau "cuda:N")."""
    requested = str(device or os.environ.get(ENV_DEVICE) or "auto").strip().lower()
    if requested in ("auto", ""):
        return "cuda" if _cuda_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if _cuda_available():
            return requested
        warnings.warn(f"device '{requested}' diminta tetapi CUDA tidak tersedia; memakai CPU.", stacklevel=2)
        return "cpu"
    raise ValueError(f"device tidak dikenal: {requested!r} (gunakan 'auto', 'cpu', 'cuda', atau 'cuda:N')")


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def onnx_cuda_available():
    """True bila onnxruntime terpasang dengan CUDAExecutionProvider (DocAligner berjalan lewat onnxruntime, bukan torch)."""
    try:
        import onnxruntime
    except ImportError:
        return False
    return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
