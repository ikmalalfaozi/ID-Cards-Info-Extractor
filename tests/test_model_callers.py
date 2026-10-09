"""Pemanggil model (P3): semua memakai model_store/manifest, tanpa jaringan dan tanpa file model sungguhan."""
import copy
import glob
import warnings
from pathlib import Path

import pytest
import torch

from src import model_store as ms

ROOT = Path(__file__).resolve().parents[1]


def import_or_skip(name):
    try:
        return __import__(name, fromlist=["_"])
    except Exception as e:  # mis. libturbojpeg tidak ada (capybara membuat TurboJPEG saat impor)
        pytest.skip(f"{name} tidak dapat diimpor di lingkungan ini: {type(e).__name__}: {e}")


@pytest.fixture
def managed(monkeypatch, tmp_path):
    """ensure_model palsu: catat nama, kembalikan berkas sementara."""
    calls = []

    def fake(name, **kw):
        calls.append(name)
        p = tmp_path / f"{name}.bin"
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(ms, "ensure_model", fake)
    return calls


class FakeYOLO:
    paths = []

    def __init__(self, path):
        FakeYOLO.paths.append(str(path))


# --------------------------------------------------------------- resolve_model_path
def test_resolve_default_uses_managed_model(managed):
    p = ms.resolve_model_path("doc-seg")
    assert managed == ["doc-seg"] and p.is_file()


def test_resolve_explicit_path_wins_and_aliases_work(managed, tmp_path):
    f = tmp_path / "mine.pt"
    f.write_bytes(b"x")
    assert ms.resolve_model_path("doc-seg", model_path=str(f)) == f
    assert ms.resolve_model_path("doc-seg", model_save_path=str(f)) == f
    assert managed == []


def test_resolve_missing_explicit_path_is_an_error_not_a_download(managed, tmp_path):
    with pytest.raises(FileNotFoundError, match="model terkelola 'doc-seg'"):
        ms.resolve_model_path("doc-seg", model_path=str(tmp_path / "tidak-ada.pt"))
    assert managed == []


def test_resolve_warns_on_deprecated_drive_id(managed):
    with pytest.warns(DeprecationWarning, match="google_drive_file_id"):
        ms.resolve_model_path("doc-seg", google_drive_file_id="ABC")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ms.resolve_model_path("doc-seg")          # tanpa argumen: tidak boleh memperingatkan


# ------------------------------------------------------------------ detektor YOLO
@pytest.mark.parametrize("module,cls,key", [("src.image_alignment.doc_detector", "DocDetector", "doc-seg"),
                                            ("src.image_alignment.orientation", "DocOrientationDetector", "doc-oc")])
def test_yolo_detectors_use_manifest_by_default(module, cls, key, managed, monkeypatch):
    mod = import_or_skip(module)
    monkeypatch.setattr(mod, "YOLO", FakeYOLO)
    FakeYOLO.paths = []
    obj = getattr(mod, cls)()
    assert managed == [key] and FakeYOLO.paths == [obj.model_path] and obj.model_path.endswith(f"{key}.bin")


@pytest.mark.parametrize("module,cls", [("src.image_alignment.doc_detector", "DocDetector"),
                                        ("src.image_alignment.orientation", "DocOrientationDetector")])
def test_yolo_detectors_keep_backward_compatible_arguments(module, cls, managed, monkeypatch, tmp_path):
    mod = import_or_skip(module)
    monkeypatch.setattr(mod, "YOLO", FakeYOLO)
    f = tmp_path / "lokal.pt"
    f.write_bytes(b"x")
    klass = getattr(mod, cls)
    assert klass(model_path=str(f)).model_path == str(f)
    assert klass(model_save_path=str(f)).model_path == str(f)           # alias lama
    with pytest.warns(DeprecationWarning):
        klass(google_drive_file_id="ID-LAMA", model_save_path=str(f))   # pemanggilan gaya lama
    assert managed == []


def test_load_doc_type_model(managed, monkeypatch):
    import src.utils as u
    monkeypatch.setattr(u, "YOLO", FakeYOLO)
    FakeYOLO.paths = []
    u.load_doc_type_model()
    assert managed == ["doc-type-cls"] and FakeYOLO.paths[0].endswith("doc-type-cls.bin")


# -------------------------------------------------------------------------- NAFNet
def tiny_opt(path_opt):
    return {"model_type": "ImageRestorationModel", "scale": 1, "num_gpu": 0,       # sengaja TANPA kunci 'dist'
            "network_g": {"type": "NAFNet", "width": 8, "middle_blk_num": 1,
                          "enc_blk_nums": [1, 1], "dec_blk_nums": [1, 1]},
            "path": path_opt}


def save_tiny_weights(tmp_path, seed):
    from src.nafnet.archs import NAFNet
    torch.manual_seed(seed)
    net = NAFNet(width=8, middle_blk_num=1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1])
    f = tmp_path / "w.pth"
    torch.save({"params": net.state_dict()}, f)
    return f, net.state_dict()


def test_nafnet_loads_managed_weights_and_no_dist_keyerror(tmp_path, monkeypatch):
    import src.nafnet.model as nm
    f, expected = save_tiny_weights(tmp_path, seed=1)
    asked = []
    monkeypatch.setattr(nm, "ensure_model", lambda name, **kw: asked.append(name) or f)
    model = nm.ImageRestorationModel(tiny_opt({"pretrain_model": "nafnet-gopro-w32", "strict_load_g": True}))
    assert asked == ["nafnet-gopro-w32"]
    for k, v in model.net_g.state_dict().items():
        assert torch.equal(v, expected[k]), k


def test_nafnet_explicit_local_path_overrides_manifest(tmp_path, monkeypatch):
    import src.nafnet.model as nm
    f, expected = save_tiny_weights(tmp_path, seed=2)
    monkeypatch.setattr(nm, "ensure_model", lambda *a, **k: pytest.fail("tidak boleh memakai manifest"))
    model = nm.ImageRestorationModel(tiny_opt({"pretrain_network_g": str(f), "pretrain_model": "nafnet-gopro-w32"}))
    assert torch.equal(next(iter(model.net_g.state_dict().values())), next(iter(expected.values())))


def test_nafnet_missing_explicit_path_is_an_error(tmp_path):
    import src.nafnet.model as nm
    with pytest.raises(FileNotFoundError, match="pretrain_model"):
        nm.ImageRestorationModel(tiny_opt({"pretrain_network_g": str(tmp_path / "x.pth")}))


def test_nafnet_without_weights_config_builds_untrained_model():
    import src.nafnet.model as nm
    assert nm.ImageRestorationModel(tiny_opt({})).net_g is not None


def test_nafnet_yaml_files_reference_manifest_keys_only():
    from src.nafnet.utils import parse
    manifest = ms.load_manifest()
    files = sorted(glob.glob(str(ROOT / "nafnet-options" / "*.yaml")))
    assert len(files) == 7
    seen = set()
    for f in files:
        path = parse(f)["path"]
        assert path.get("pretrain_model") in manifest["models"], f
        assert "pretrain_network_g" not in path and "pretrain_network_g_gdrive_id" not in path, f
        seen.add(path["pretrain_model"])
    assert seen == {k for k in manifest["models"] if k.startswith(("nafnet", "nafssr"))}


# --------------------------------------------------------------------------- Donut
class FakePretrained:
    calls = []

    @classmethod
    def from_pretrained(cls, name, **kw):
        FakePretrained.calls.append((name, kw))
        return object()


@pytest.fixture
def donut(monkeypatch):
    import src.ocr.donut as d
    monkeypatch.setattr(d, "DonutProcessor", type("P", (FakePretrained,), {}))
    monkeypatch.setattr(d, "VisionEncoderDecoderModel", type("M", (FakePretrained,), {}))
    FakePretrained.calls = []
    return d


def test_donut_default_is_pinned_to_manifest_commit(donut):
    repo, rev = ms.external_revision("donut")
    ex = donut.DonutInfoExtractor()
    assert ex.model_name == repo and ex.revision == rev
    assert [c[0] for c in FakePretrained.calls] == [repo, repo]
    assert all(c[1] == {"revision": rev, "cache_dir": None} for c in FakePretrained.calls)


def test_donut_old_style_call_with_default_name_is_still_pinned(donut):
    repo, rev = ms.external_revision("donut")
    assert donut.DonutInfoExtractor(repo).revision == rev


def test_donut_other_model_is_not_pinned_unless_asked(donut):
    assert donut.DonutInfoExtractor("orang/lain").revision is None
    assert donut.DonutInfoExtractor("orang/lain", revision="abc").revision == "abc"


def test_donut_no_longer_writes_into_repo_models_dir(donut, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    donut.DonutInfoExtractor()
    assert not (tmp_path / "models").exists()
