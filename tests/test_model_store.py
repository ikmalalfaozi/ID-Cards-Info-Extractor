"""Pengujian src/model_store.py tanpa jaringan (hf_hub_download dan gdown diganti versi palsu)."""
import copy
import hashlib
import os
from pathlib import Path

import pytest

from src import model_store as ms

CONTENT = b"bobot-palsu-untuk-uji"
SHA = hashlib.sha256(CONTENT).hexdigest()
REV = "a" * 40


def make_manifest(revision=REV, sha=SHA, gdrive="GID123"):
    entry = {"file": "yolo/fake.pt", "sha256": sha, "size": len(CONTENT)}
    if gdrive:
        entry["fallback"] = {"gdrive": gdrive}
    return {"schema": 1,
            "hub": {"repo_id": "user/repo", "tag": "v1.0", "revision": revision},
            "models": {"fake": entry},
            "external": {"donut": {"repo_id": "user/donut", "revision": "b" * 40}}}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("IDCARD_MODELS_DIR", "IDCARD_NO_FALLBACK", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IDCARD_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """Pasang hf_hub_download palsu; kembalikan list panggilan."""
    import huggingface_hub
    calls = []

    def fake(**kw):
        calls.append(kw)
        p = tmp_path / "hub" / kw["filename"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(CONTENT)
        return str(p)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return calls


@pytest.fixture
def fake_gdown(monkeypatch):
    import gdown
    calls = []

    def fake(id=None, output=None, quiet=False, **kw):
        calls.append(id)
        Path(output).write_bytes(CONTENT)
        return output

    monkeypatch.setattr(gdown, "download", fake)
    return calls


def break_hub(monkeypatch, exc=ConnectionError("tidak ada jaringan")):
    import huggingface_hub

    def boom(**kw):
        raise exc

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)


# --------------------------------------------------------------- manifest asli
def test_real_manifest_loads_and_has_expected_models():
    m = ms.load_manifest()
    assert {"doc-seg", "doc-oc", "doc-type-cls"} <= set(m["models"])
    assert len([k for k in m["models"] if k.startswith(("nafnet", "nafssr"))]) == 7


def test_real_manifest_formats_valid_when_filled():
    m = ms.load_manifest()
    for name, e in m["models"].items():
        if e.get("sha256"):
            assert ms._SHA256_RE.match(e["sha256"]), name
    if m["hub"].get("revision"):
        assert ms._COMMIT_RE.match(m["hub"]["revision"])
    files = [e["file"] for e in m["models"].values()]
    assert len(files) == len(set(files))


def test_real_manifest_yolo_checksums_match_local_files():
    root = Path(__file__).resolve().parents[1] / "models"
    m = ms.load_manifest()
    checked = 0
    for name in ("doc-seg", "doc-oc", "doc-type-cls"):
        local = root / Path(m["models"][name]["file"]).name
        if local.is_file():
            assert ms.sha256_file(local) == m["models"][name]["sha256"], name
            checked += 1
    if not checked:
        pytest.skip("file model lokal tidak ada")


def test_external_revision_donut_is_pinned_commit():
    repo, rev = ms.external_revision("donut")
    assert repo.startswith("ikmalalfaozi/donut") and ms._COMMIT_RE.match(rev)


def test_manifest_issues_reports_unfinished_release():
    issues = ms.manifest_issues()           # manifest asli belum dipublikasikan
    assert any("hub.revision" in i for i in issues)
    assert any("nafnet-gopro-w32: sha256 kosong" in i for i in issues)
    assert ms.manifest_issues(make_manifest()) == []


# ------------------------------------------------------------------ ensure_model
def test_hub_download_uses_pinned_revision_and_verifies(fake_hub):
    p = ms.ensure_model("fake", manifest=make_manifest())
    assert p.read_bytes() == CONTENT
    assert fake_hub == [dict(repo_id="user/repo", filename="yolo/fake.pt", revision=REV, local_files_only=False)]


def test_checksum_mismatch_raises_and_never_falls_back(fake_hub, fake_gdown):
    with pytest.raises(ms.ChecksumMismatchError):
        ms.ensure_model("fake", manifest=make_manifest(sha="0" * 64))
    assert fake_gdown == []


def test_verify_false_skips_checksum(fake_hub):
    assert ms.ensure_model("fake", manifest=make_manifest(sha="0" * 64), verify=False).is_file()


def test_missing_sha_warns_but_works(fake_hub):
    with pytest.warns(UserWarning, match="sha256 belum diisi"):
        ms.ensure_model("fake", manifest=make_manifest(sha=None))


def test_offline_uses_cache_only_and_no_fallback(monkeypatch, fake_gdown):
    import huggingface_hub
    seen = []

    def offline_miss(**kw):
        seen.append(kw["local_files_only"])
        raise huggingface_hub.errors.LocalEntryNotFoundError("tidak ada di cache")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline_miss)
    with pytest.raises(ms.ModelUnavailableError, match="offline"):
        ms.ensure_model("fake", manifest=make_manifest(), offline=True)
    assert seen == [True] and fake_gdown == []


def test_hf_hub_offline_env_is_respected(monkeypatch, fake_hub):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    ms.ensure_model("fake", manifest=make_manifest())
    assert fake_hub[0]["local_files_only"] is True


def test_hub_failure_falls_back_to_gdown_and_verifies(monkeypatch, fake_gdown):
    break_hub(monkeypatch)
    p = ms.ensure_model("fake", manifest=make_manifest())
    assert fake_gdown == ["GID123"] and p.read_bytes() == CONTENT
    assert SHA in p.parts                      # disimpan di <cache>/<sha256>/<nama>


def test_gdown_result_is_checksum_verified(monkeypatch, fake_gdown):
    break_hub(monkeypatch)
    with pytest.raises(ms.ChecksumMismatchError):
        ms.ensure_model("fake", manifest=make_manifest(sha="0" * 64))


def test_no_fallback_env_disables_gdown(monkeypatch, fake_gdown):
    break_hub(monkeypatch)
    monkeypatch.setenv("IDCARD_NO_FALLBACK", "1")
    with pytest.raises(ms.ModelUnavailableError, match="IDCARD_NO_FALLBACK"):
        ms.ensure_model("fake", manifest=make_manifest())
    assert fake_gdown == []


def test_unpublished_manifest_skips_hub_and_uses_fallback(fake_hub, fake_gdown):
    p = ms.ensure_model("fake", manifest=make_manifest(revision=None))
    assert fake_hub == [] and fake_gdown == ["GID123"] and p.is_file()


def test_unpublished_without_fallback_gives_actionable_error(fake_hub, fake_gdown):
    with pytest.raises(ms.ModelUnavailableError) as ei:
        ms.ensure_model("fake", manifest=make_manifest(revision=None, gdrive=None))
    msg = str(ei.value)
    assert "belum dipublikasikan" in msg and "IDCARD_MODELS_DIR" in msg


def test_failed_gdown_leaves_no_partial_file(monkeypatch, tmp_path):
    import gdown
    break_hub(monkeypatch)

    def partial(id=None, output=None, **kw):
        Path(output).write_bytes(b"parsial")
        raise RuntimeError("kuota terlampaui")

    monkeypatch.setattr(gdown, "download", partial)
    with pytest.raises(ms.ModelUnavailableError, match="kuota"):
        ms.ensure_model("fake", manifest=make_manifest())
    assert not list((tmp_path / "cache").rglob("fake.pt"))


def test_mirror_dir_takes_priority_and_is_verified(monkeypatch, tmp_path, fake_hub):
    mirror = tmp_path / "mirror"
    (mirror / "yolo").mkdir(parents=True)
    (mirror / "yolo" / "fake.pt").write_bytes(CONTENT)
    monkeypatch.setenv("IDCARD_MODELS_DIR", str(mirror))
    p = ms.ensure_model("fake", manifest=make_manifest())
    assert p == mirror / "yolo" / "fake.pt" and fake_hub == []
    (mirror / "yolo" / "fake.pt").write_bytes(b"diubah")
    with pytest.raises(ms.ChecksumMismatchError):
        ms.ensure_model("fake", manifest=make_manifest())


def test_unknown_model_lists_available():
    with pytest.raises(KeyError, match="doc-seg"):
        ms.ensure_model("tidak-ada")


def test_verify_all_reports_per_model(fake_hub):
    m = make_manifest()
    m["models"]["rusak"] = {**copy.deepcopy(m["models"]["fake"]), "file": "yolo/rusak.pt", "sha256": "0" * 64}
    res = ms.verify_all(manifest=m)
    assert res["fake"] == "ok" and res["rusak"].startswith("ChecksumMismatchError")
