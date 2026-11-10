"""CornerModel: pilihan model_type/model_cfg, pasca-pemrosesan 'point', dan pendaftaran model di manifest."""
import numpy as np
import pytest

from src import model_store
from src.image_alignment import corner_model as cm


def test_defaults_and_enum_like_types():
    assert cm._normalize("heatmap", None) == ("heatmap", "fastvit_sa24")
    assert cm._normalize("point", None) == ("point", "lcnet050")
    assert cm._normalize("HEATMAP", "lcnet100") == ("heatmap", "lcnet100")

    class ModelType:           # mirip docaligner.ModelType.heatmap
        name = "heatmap"
    assert cm._normalize(ModelType(), "fastvit_t8") == ("heatmap", "fastvit_t8")


def test_invalid_type_or_cfg_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model type"):
        cm._normalize("segmentation", None)
    with pytest.raises(ValueError, match="Invalid model_cfg"):
        cm._normalize("point", "fastvit_sa24")


def test_every_variant_is_registered_in_manifest_with_checksum():
    models = model_store.load_manifest()["models"]
    for cfgs in cm.MODELS.values():
        for name in cfgs.values():
            assert name in models, name
            assert len(models[name]["sha256"]) == 64 and models[name]["size"] > 0
            assert models[name]["file"].startswith("docaligner/")


def test_point_postprocess_scales_to_pixels_and_handles_no_document():
    pts = np.array([[0.1, 0.2, 0.9, 0.2, 0.9, 0.8, 0.1, 0.8]], dtype="float32")
    out = cm._corners_from_points(pts, np.array([[0.9]]), (100, 200))      # (tinggi, lebar)
    assert np.allclose(out, [[20, 20], [180, 20], [180, 80], [20, 80]])
    assert cm._corners_from_points(pts, np.array([[0.1]]), (100, 200)).size == 0


def test_heatmap_postprocess_returns_centroid_of_largest_blob():
    channel = np.zeros((64, 64), dtype="float32")
    channel[10:20, 10:20] = 1.0          # blob besar
    channel[40:43, 40:43] = 1.0          # blob kecil, harus diabaikan
    x, y = cm._corner_from_heatmap(channel, (64, 64))
    assert (round(x), round(y)) == (14, 14)
    assert cm._corner_from_heatmap(np.zeros((64, 64), dtype="float32"), (64, 64)) is None


# ------------------------------------------------------------------ backend torch
def _tiny_onnx(path):
    """Model mini: img (1,3,256,256) -> Sigmoid -> heatmap; cukup untuk menguji jalur ort vs torch tanpa unduhan."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    node = helper.make_node("Sigmoid", ["img"], ["heatmap"])
    graph = helper.make_graph([node], "tiny", [helper.make_tensor_value_info("img", TensorProto.FLOAT, [1, 3, 256, 256])],
                              [helper.make_tensor_value_info("heatmap", TensorProto.FLOAT, [1, 3, 256, 256])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return str(path)


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported backend"):
        cm.CornerModel(backend="tensorrt")


def test_torch_backend_gives_actionable_error_without_onnx2torch():
    import subprocess
    import sys
    import textwrap
    from pathlib import Path
    code = textwrap.dedent("""
        import sys
        class Block:
            def find_spec(self, name, path=None, target=None):
                if name.split('.')[0] == 'onnx2torch':
                    raise ImportError('diblokir untuk uji')
        sys.meta_path.insert(0, Block())
        from src.image_alignment.corner_model import _TorchSession
        try:
            _TorchSession('x.onnx', 'cpu')
        except ImportError as e:
            print('ImportError:', e)
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=Path(__file__).resolve().parents[1])
    assert "ImportError: Backend 'torch' membutuhkan onnx2torch" in r.stdout and "torch-backend.txt" in r.stdout, \
        r.stdout + r.stderr[-400:]


def test_torch_backend_matches_onnx_backend(tmp_path):
    pytest.importorskip("onnx2torch")
    pytest.importorskip("onnxruntime")
    path = _tiny_onnx(tmp_path / "tiny.onnx")
    img = np.zeros((120, 200, 3), dtype="uint8")
    for (y, x) in ((20, 20), (20, 170), (100, 170), (100, 20)):      # empat blob terang di sudut
        img[y - 8:y + 8, x - 8:x + 8] = 255
    a = cm.CornerModel("heatmap", None, model_path=path, device="cpu", backend="onnx")(img)
    b = cm.CornerModel("heatmap", None, model_path=path, device="cpu", backend="torch")(img)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.allclose(a, b, atol=1e-3)
