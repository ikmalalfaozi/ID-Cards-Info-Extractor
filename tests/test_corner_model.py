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
