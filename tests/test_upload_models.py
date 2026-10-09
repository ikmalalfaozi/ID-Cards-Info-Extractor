"""Pengujian scripts/upload_models.py dengan HfApi palsu (tanpa jaringan, tanpa token)."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import model_store as ms

SPEC = importlib.util.spec_from_file_location("upload_models", Path(__file__).resolve().parents[1] / "scripts" / "upload_models.py")
um = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(um)

OID = "c" * 40


class FakeApi:
    def __init__(self, tags=(), whoami_ok=True, corrupt_remote=False, no_lfs=False):
        self.tags, self.calls, self.whoami_ok = set(tags), [], whoami_ok
        self.corrupt_remote, self.no_lfs, self.uploaded = corrupt_remote, no_lfs, {}

    def whoami(self):
        if not self.whoami_ok:
            raise RuntimeError("token tidak ada")
        return {"name": "ikmalalfaozi"}

    def create_repo(self, **kw):
        self.calls.append(("create_repo", kw))

    def list_repo_refs(self, **kw):
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in self.tags])

    def create_commit(self, **kw):
        self.calls.append(("create_commit", kw))
        for op in kw["operations"]:
            self.uploaded[op.path_in_repo] = op.path_or_fileobj
        return SimpleNamespace(oid=OID)

    def create_tag(self, **kw):
        self.calls.append(("create_tag", kw))

    def get_paths_info(self, repo_id, paths, revision, repo_type):
        out = []
        for p in paths:
            src = self.uploaded[p]
            sha = ms.sha256_file(Path(src)) if isinstance(src, str) else None
            if self.corrupt_remote:
                sha = "0" * 64
            out.append(SimpleNamespace(path=p, lfs=None if self.no_lfs else {"sha256": sha}))
        return out

    def names(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Manifest mini + berkas lokal palsu + MANIFEST_PATH sementara."""
    yolo, work = tmp_path / "yolo", tmp_path / "work"
    yolo.mkdir(), work.mkdir()
    (yolo / "a.pt").write_bytes(b"model-a")
    (work / "b.pth").write_bytes(b"model-b")
    manifest = {"schema": 1,
                "hub": {"repo_id": "user/repo", "tag": "v1.0", "revision": None},
                "models": {"a": {"file": "yolo/a.pt", "sha256": ms.sha256_file(yolo / "a.pt"), "size": 7},
                           "b": {"file": "nafnet/b.pth", "sha256": None, "size": None, "fallback": {"gdrive": "GB"}}},
                "external": {"donut": {"repo_id": "u/d", "revision": "b" * 40}}}
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest))
    monkeypatch.setattr(ms, "MANIFEST_PATH", mpath)
    base = ["--yolo-dir", str(yolo), "--workdir", str(work)]
    return SimpleNamespace(yolo=yolo, work=work, mpath=mpath, base=base, manifest=manifest)


def run(env, extra, api=None, answer="unggah"):
    return um.main(env.base + extra, api=api, license_fetcher=lambda: "TEKS LISENSI", confirm=lambda _: answer)


def test_dry_run_changes_nothing(env, capsys):
    api = FakeApi()
    assert run(env, [], api=api) == 0
    assert api.calls == [] and json.loads(env.mpath.read_text()) == env.manifest
    out = capsys.readouterr().out
    assert "Dry-run" in out and "yolo/a.pt" in out


def test_dry_run_reports_missing_and_does_not_download(env, capsys, monkeypatch):
    (env.work / "b.pth").unlink()
    import gdown
    monkeypatch.setattr(gdown, "download", lambda **kw: pytest.fail("tidak boleh mengunduh tanpa --download"))
    assert run(env, []) == 0
    assert "missing" in capsys.readouterr().out


def test_download_fetches_missing_from_drive(env, monkeypatch):
    (env.work / "b.pth").unlink()
    import gdown
    got = []

    def fake(id=None, output=None, quiet=False, **kw):
        got.append(id)
        Path(output).write_bytes(b"model-b")
        return output

    monkeypatch.setattr(gdown, "download", fake)
    assert run(env, ["--download"]) == 0
    assert got == ["GB"] and (env.work / "b.pth").is_file()


def test_failed_download_leaves_no_partial_file(env, monkeypatch):
    (env.work / "b.pth").unlink()
    import gdown

    def partial(id=None, output=None, **kw):
        Path(output).write_bytes(b"parsial")
        raise RuntimeError("kuota terlampaui")

    monkeypatch.setattr(gdown, "download", partial)
    run(env, ["--download"])
    assert not (env.work / "b.pth").exists()


def test_changed_local_file_is_refused_without_flag(env):
    (env.yolo / "a.pt").write_bytes(b"model-a-BARU")
    api = FakeApi()
    assert run(env, ["--execute", "--license", "mit", "--yes"], api=api) == 2
    assert api.calls == []
    assert run(env, ["--execute", "--license", "mit", "--yes", "--accept-changes"], api=FakeApi()) == 0


def test_execute_requires_license(env):
    api = FakeApi()
    assert run(env, ["--execute", "--yes"], api=api) == 2 and api.calls == []


def test_execute_refuses_when_files_missing(env):
    (env.work / "b.pth").unlink()
    api = FakeApi()
    assert run(env, ["--execute", "--license", "mit", "--yes"], api=api) == 2 and api.calls == []


def test_execute_requires_login(env):
    api = FakeApi(whoami_ok=False)
    assert run(env, ["--execute", "--license", "mit", "--yes"], api=api) == 2 and api.calls == []


def test_confirmation_must_be_exact(env):
    api = FakeApi()
    assert run(env, ["--execute", "--license", "mit"], api=api, answer="ya") == 1 and api.calls == []


def test_execute_full_flow_updates_manifest(env):
    api = FakeApi()
    assert run(env, ["--execute", "--license", "agpl-3.0", "--yes", "--tag", "v1.0"], api=api) == 0
    assert api.names() == ["create_repo", "create_commit", "create_tag"]
    repo_kw = api.calls[0][1]
    assert repo_kw["private"] is False and repo_kw["exist_ok"] is True
    assert api.calls[2][1]["tag"] == "v1.0" and api.calls[2][1]["revision"] == OID
    assert set(api.uploaded) == {"yolo/a.pt", "nafnet/b.pth", "README.md", "LICENSES/NAFNet-and-BasicSR-LICENSE.txt"}
    new = json.loads(env.mpath.read_text())
    assert new["hub"]["revision"] == OID and new["hub"]["tag"] == "v1.0"
    assert new["models"]["b"]["sha256"] == ms.sha256_file(env.work / "b.pth") and new["models"]["b"]["size"] == 7
    assert new["external"] == env.manifest["external"]            # bagian lain tidak berubah
    assert ms.manifest_issues(new) == []


def test_model_card_has_license_checksums_and_attribution(env):
    api = FakeApi()
    run(env, ["--execute", "--license", "agpl-3.0", "--yes"], api=api)
    card = api.uploaded["README.md"].decode()
    assert card.startswith("---\nlicense: agpl-3.0\n")
    assert ms.sha256_file(env.yolo / "a.pt") in card and "megvii-research/NAFNet" in card and "MIT" in card
    assert api.uploaded["LICENSES/NAFNet-and-BasicSR-LICENSE.txt"] == b"TEKS LISENSI"


def test_existing_tag_is_refused_before_any_commit(env):
    api = FakeApi(tags={"v1.0"})
    with pytest.raises(SystemExit, match="v1.0"):
        run(env, ["--execute", "--license", "mit", "--yes", "--tag", "v1.0"], api=api)
    assert "create_commit" not in api.names() and json.loads(env.mpath.read_text()) == env.manifest


def test_remote_checksum_mismatch_keeps_manifest_unchanged(env, capsys):
    api = FakeApi(corrupt_remote=True)
    assert run(env, ["--execute", "--license", "mit", "--yes"], api=api) == 3
    assert json.loads(env.mpath.read_text()) == env.manifest
    assert "GAGAL" in capsys.readouterr().out


def test_unverifiable_remote_file_is_treated_as_failure(env):
    api = FakeApi(no_lfs=True)
    assert run(env, ["--execute", "--license", "mit", "--yes"], api=api) == 3
    assert json.loads(env.mpath.read_text()) == env.manifest


def test_only_subset_updates_only_those_models(env):
    api = FakeApi()
    assert run(env, ["--execute", "--license", "mit", "--yes", "--only", "b"], api=api) == 0
    assert "yolo/a.pt" not in api.uploaded
    new = json.loads(env.mpath.read_text())
    assert new["models"]["b"]["sha256"] and new["models"]["a"] == env.manifest["models"]["a"]


def test_unknown_only_key_is_rejected(env):
    with pytest.raises(SystemExit, match="tidak ada"):
        run(env, ["--only", "zzz"])


def test_license_fetch_uses_nafnet_url():
    seen = []

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return b"MIT License ..."

    def opener(url, timeout):
        seen.append(url)
        return R()

    assert um.fetch_license_text(opener) == "MIT License ..." and seen == [um.NAFNET_LICENSE_URL]


# ------------------------------------------------------------ license: other
def test_other_requires_yolo_license_and_has_no_default(env):
    api = FakeApi()
    assert run(env, ["--execute", "--license", "other", "--yes"], api=api) == 2 and api.calls == []


def test_other_uploads_license_md_and_front_matter(env):
    api = FakeApi()
    assert run(env, ["--execute", "--license", "other", "--yolo-license", "AGPL-3.0", "--yes"], api=api) == 0
    assert "LICENSE.md" in api.uploaded
    card = api.uploaded["README.md"].decode()
    assert card.startswith("---\nlicense: other\nlicense_name: per-file-license\nlicense_link: LICENSE.md\ntags:")
    assert "[LICENSE.md](LICENSE.md)" in card
    lic = api.uploaded["LICENSE.md"].decode()
    assert "AGPL-3.0" in lic and "`yolo/*.pt`" in lic and "MIT" in lic and "megvii-research/NAFNet" in lic


def test_other_license_link_is_optional_and_rendered(env):
    api = FakeApi()
    run(env, ["--execute", "--license", "other", "--yolo-license", "AGPL-3.0", "--yolo-license-link",
              "https://www.gnu.org/licenses/agpl-3.0.html", "--yes"], api=api)
    assert "[teks lisensi](https://www.gnu.org/licenses/agpl-3.0.html)" in api.uploaded["LICENSE.md"].decode()


def test_other_license_md_lists_only_present_groups(env):
    rows = [{"key": "b", "file": "nafnet/b.pth", "size": 1, "sha256": "x"}]
    lic = um.build_license_md(rows, "AGPL-3.0")
    assert "nafnet" in lic and "yolo/*.pt" not in lic


def test_single_license_has_no_license_md_and_no_other_fields(env):
    api = FakeApi()
    assert run(env, ["--execute", "--license", "agpl-3.0", "--yes"], api=api) == 0
    assert "LICENSE.md" not in api.uploaded
    card = api.uploaded["README.md"].decode()
    assert "license_name" not in card and "## Lisensi" not in card


def test_license_other_does_not_change_manifest_flow(env):
    api = FakeApi()
    run(env, ["--execute", "--license", "other", "--yolo-license", "AGPL-3.0", "--yes"], api=api)
    new = json.loads(env.mpath.read_text())
    assert new["hub"]["revision"] == OID and ms.manifest_issues(new) == []
