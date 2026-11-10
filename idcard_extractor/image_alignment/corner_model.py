"""Deteksi 4 sudut dokumen dengan model DocAligner (heatmap/point), tanpa paket docaligner/capybara.

Hanya butuh onnxruntime, cv2 dan numpy. Pra/pasca-pemrosesan sama dengan `docaligner.heatmap_reg.infer` dan
`docaligner.point_reg.infer` (heatmap: ambang 0.3 -> Otsu -> kontur terluar terbesar -> centroid), supaya
hasilnya identik.
"""
import cv2
import numpy as np

from ..config import get_device, onnx_cuda_available
from ..model_store import resolve_model_path

_INPUT_SIZE = (256, 256)  # (tinggi, lebar); sama untuk semua varian
_HEATMAP_THRESHOLD = 0.3

# model_type -> {model_cfg: nama di idcard_extractor/model_manifest.json}; elemen pertama = bawaan
MODELS = {
    "heatmap": {
        "fastvit_sa24": "doc-corner-heatmap-fastvit-sa24",
        "fastvit_t8": "doc-corner-heatmap-fastvit-t8",
        "lcnet100": "doc-corner-heatmap-lcnet100",
    },
    "point": {
        "lcnet050": "doc-corner-point-lcnet050",
    },
}


def _normalize(model_type, model_cfg):
    """Terima str atau enum (mis. docaligner.ModelType); kembalikan (model_type, model_cfg) yang valid."""
    model_type = str(getattr(model_type, "name", model_type)).lower()
    if model_type not in MODELS:
        raise ValueError(f"Unsupported model type: {model_type!r} (use {' or '.join(map(repr, MODELS))}).")
    available = MODELS[model_type]
    model_cfg = model_cfg or next(iter(available))
    if model_cfg not in available:
        raise ValueError(f"Invalid model_cfg {model_cfg!r} for model_type {model_type!r}; valid: {list(available)}")
    return model_type, model_cfg


def _corners_from_points(points, has_obj, size_hw):
    """Varian 'point': 8 nilai ternormalisasi -> 4 titik piksel; kosong bila model menilai tak ada dokumen."""
    if has_obj > 0.5:
        return points.reshape(4, 2) * np.array(size_hw[::-1])
    return np.array([])


def _corner_from_heatmap(channel, size_hw):
    """Satu titik dari satu channel heatmap: kontur terluar dengan luas terbesar, titik beratnya. None bila kosong."""
    pred = cv2.resize(channel, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_LINEAR)
    pred[pred < _HEATMAP_THRESHOLD] = 0
    binary = cv2.threshold(np.uint8(pred * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = [(cv2.moments(c), c) for c in contours if c.shape[0] > 1]
    moments = [m for m, _ in moments if m["m00"] > 0]  # drop_empty
    if not moments:
        return None
    m = max(moments, key=lambda m: m["m00"])
    return [m["m10"] / (m["m00"] + 1e-5), m["m01"] / (m["m00"] + 1e-5)]


BACKENDS = ("onnx", "torch")


class _TorchSession:
    """Menjalankan grafik ONNX sebagai modul torch (onnx2torch); antarmukanya meniru `onnxruntime.InferenceSession.run`."""

    def __init__(self, model_path, device):
        try:
            import torch
            from onnx2torch import convert
        except ImportError as e:
            raise ImportError("Backend 'torch' membutuhkan onnx2torch. Pasang dengan: "
                              "pip install -r requirements/torch-backend.txt") from e
        self._torch = torch
        self.device = device
        self.module = convert(model_path).eval().to(device)

    def run(self, output_names, feed):
        torch = self._torch
        with torch.inference_mode():
            out = self.module(torch.from_numpy(feed["img"]).to(self.device))
        out = list(out) if isinstance(out, (list, tuple)) else [out]
        return [o.cpu().numpy() for o in out]


def _onnx_session(model_path, device):
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime tidak terpasang. Pasang dengan: pip install onnxruntime "
                          "(atau onnxruntime-gpu untuk CUDA).") from e

    use_cuda = device.startswith("cuda") and onnx_cuda_available()
    if device.startswith("cuda") and not use_cuda:
        print("CornerModel: onnxruntime-gpu tidak terpasang. Memakai CPU.")
    else:
        print(f"CornerModel: using {'GPU (CUDA)' if use_cuda else 'CPU'}.")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 2
    if use_cuda:
        device_id = int(device.split(":")[1]) if ":" in device else 0
        providers = [("CUDAExecutionProvider", {"device_id": device_id, "cudnn_conv_use_max_workspace": "1"})]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, sess_options=opts, providers=providers)


class CornerModel:
    def __init__(self, model_type="heatmap", model_cfg=None, model_path=None, device=None, backend="onnx"):
        """backend: "onnx" (onnxruntime, bawaan) atau "torch" (onnx2torch; opsional, lebih lambat di CPU)."""
        if backend not in BACKENDS:
            raise ValueError(f"Unsupported backend: {backend!r} (use {' or '.join(map(repr, BACKENDS))}).")
        self.backend = backend
        self.model_type, self.model_cfg = _normalize(model_type, model_cfg)
        self.device = get_device(device)
        name = MODELS[self.model_type][self.model_cfg]
        self.model_path = str(resolve_model_path(name, model_path=model_path))
        if backend == "torch":
            print(f"CornerModel[torch]: using {self.device}.")
            self.session = _TorchSession(self.model_path, self.device)
        else:
            self.session = _onnx_session(self.model_path, self.device)

    def __call__(self, image):
        """Kembalikan array float32 (n, 2) berisi titik (x, y); n < 4 bila ada sudut yang tak terdeteksi."""
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("Input image must be a numpy array (H, W, 3).")
        height, width = image.shape[:2]
        x = cv2.resize(image, (_INPUT_SIZE[1], _INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        x = np.transpose(x, (2, 0, 1)).astype("float32")[None] / 255.0
        if self.model_type == "point":
            points, has_obj = self.session.run(["points", "has_obj"], {"img": x})
            return _corners_from_points(points, has_obj, (height, width))
        heatmap = self.session.run(["heatmap"], {"img": x})[0]  # (1, 4, H, W)

        points = []
        for index, channel in enumerate(heatmap[0]):
            point = _corner_from_heatmap(channel, (height, width))
            if point is not None and index < 4:
                points.append(point)
        return np.array(points).astype(np.float32)
