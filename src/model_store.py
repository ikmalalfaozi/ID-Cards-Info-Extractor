"""Penyimpanan dan pemuatan model berdasarkan manifest (lihat docs/model-storage-design.md).

    path = ensure_model("doc-seg")          # Path lokal; mengunduh bila belum ada

Urutan sumber:
  1. mirror lokal      $IDCARD_MODELS_DIR/<file>
  2. Hugging Face Hub  repo + commit sha dari manifest (cache bawaan huggingface_hub)
  3. gdown (cadangan)  hanya bila online, ada ID Drive, dan IDCARD_NO_FALLBACK tidak di-set

Checksum yang tidak cocok selalu menjadi error (ChecksumMismatchError), tanpa fallback.

Variabel lingkungan: IDCARD_MODELS_DIR, IDCARD_CACHE_DIR, IDCARD_NO_FALLBACK, HF_HUB_OFFLINE.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MANIFEST_PATH = Path(__file__).resolve().parent / "model_manifest.json"

_TRUE = {"1", "true", "yes", "on"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class ModelUnavailableError(RuntimeError):
    """Model tidak dapat diperoleh dari sumber mana pun."""


class ChecksumMismatchError(RuntimeError):
    """Isi file tidak sama dengan sha256 di manifest (file rusak atau berubah)."""


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    with open(path or MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def _entry(manifest: Dict[str, Any], name: str) -> Dict[str, Any]:
    models = manifest.get("models", {})
    if name not in models:
        raise KeyError(f"Model '{name}' tidak ada di manifest. Tersedia: {', '.join(sorted(models))}")
    return models[name]


def _cache_root() -> Path:
    return Path(os.environ.get("IDCARD_CACHE_DIR") or Path.home() / ".cache" / "idcard_extractor")


def _verify(path: Path, name: str, entry: Dict[str, Any], verify: bool) -> None:
    if not verify:
        return
    expected = entry.get("sha256")
    if not expected:
        warnings.warn(f"Model '{name}': sha256 belum diisi di manifest, integritas file tidak diverifikasi.",
                      stacklevel=3)
        return
    actual = sha256_file(path)
    if actual != expected:
        raise ChecksumMismatchError(
            f"Model '{name}' ({path}) tidak cocok dengan manifest.\n"
            f"  diharapkan: {expected}\n  diperoleh : {actual}\n"
            f"Hapus file/cache tersebut lalu coba lagi; jangan memuat file yang berubah."
        )


def _from_mirror(name: str, entry: Dict[str, Any], verify: bool) -> Optional[Path]:
    root = os.environ.get("IDCARD_MODELS_DIR")
    if not root:
        return None
    path = Path(root) / entry["file"]
    if not path.is_file():
        return None
    _verify(path, name, entry, verify)
    return path


def _from_hub(manifest: Dict[str, Any], name: str, entry: Dict[str, Any], offline: bool,
              errors: List[str]) -> Optional[Path]:
    hub = manifest.get("hub", {})
    if not hub.get("revision"):
        errors.append("Hugging Face: manifest belum dipublikasikan (hub.revision kosong; jalankan scripts/upload_models.py)")
        return None
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id=hub["repo_id"], filename=entry["file"],
                                    revision=hub["revision"], local_files_only=offline))
    except Exception as e:  # jaringan, 404, cache kosong saat offline, dsb.
        errors.append(f"Hugging Face ({hub['repo_id']}@{hub['revision'][:8]}): {type(e).__name__}: {e}")
        return None


def _from_gdrive(name: str, entry: Dict[str, Any], errors: List[str]) -> Optional[Path]:
    gid = (entry.get("fallback") or {}).get("gdrive")
    if _env_true("IDCARD_NO_FALLBACK"):
        errors.append("gdown: dinonaktifkan oleh IDCARD_NO_FALLBACK")
        return None
    if not gid:
        errors.append("gdown: tidak ada ID Drive untuk model ini")
        return None
    dest = _cache_root() / (entry.get("sha256") or "unverified") / Path(entry["file"]).name
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown

        out = gdown.download(id=gid, output=str(dest), quiet=False)
    except Exception as e:
        dest.unlink(missing_ok=True)  # jangan tinggalkan file parsial
        errors.append(f"gdown (id={gid}): {type(e).__name__}: {e}")
        return None
    if not out or not dest.is_file():
        dest.unlink(missing_ok=True)
        errors.append(f"gdown (id={gid}): unduhan gagal (kuota Drive atau file tidak dapat diakses)")
        return None
    return dest


def ensure_model(name: str, *, offline: Optional[bool] = None, verify: bool = True,
                 manifest: Optional[Dict[str, Any]] = None) -> Path:
    """Kembalikan path lokal model `name`, mengunduh bila perlu.

    offline: True -> hanya mirror/cache. Default: ikuti HF_HUB_OFFLINE.
    verify : cocokkan sha256 dengan manifest.
    manifest: untuk pengujian; default manifest di repo.
    """
    manifest = manifest if manifest is not None else load_manifest()
    entry = _entry(manifest, name)
    offline = _env_true("HF_HUB_OFFLINE") if offline is None else offline

    path = _from_mirror(name, entry, verify)
    if path:
        return path

    errors: List[str] = []
    path = _from_hub(manifest, name, entry, offline, errors)
    if path:
        _verify(path, name, entry, verify)
        return path

    if offline:
        raise ModelUnavailableError(
            f"Model '{name}' tidak ada di cache dan mode offline aktif.\n  - " + "\n  - ".join(errors)
            + "\nJalankan sekali dengan internet, atau letakkan file di $IDCARD_MODELS_DIR/" + entry["file"])

    path = _from_gdrive(name, entry, errors)
    if path:
        _verify(path, name, entry, verify)
        return path

    raise ModelUnavailableError(
        f"Model '{name}' tidak dapat diperoleh dari sumber mana pun:\n  - " + "\n  - ".join(errors)
        + "\nAlternatif: letakkan file di $IDCARD_MODELS_DIR/" + entry["file"])


def external_revision(name: str, manifest: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """(repo_id, revision) untuk model eksternal di Hugging Face, mis. 'donut'."""
    manifest = manifest if manifest is not None else load_manifest()
    ext = manifest.get("external", {})
    if name not in ext:
        raise KeyError(f"Model eksternal '{name}' tidak ada di manifest. Tersedia: {', '.join(sorted(ext))}")
    return ext[name]["repo_id"], ext[name]["revision"]


def list_models(manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    manifest = manifest if manifest is not None else load_manifest()
    return dict(manifest.get("models", {}))


def manifest_issues(manifest: Optional[Dict[str, Any]] = None) -> List[str]:
    """Daftar masalah format/kelengkapan manifest (kosong = siap rilis)."""
    manifest = manifest if manifest is not None else load_manifest()
    issues: List[str] = []
    hub = manifest.get("hub", {})
    if not hub.get("repo_id"):
        issues.append("hub.repo_id kosong")
    rev = hub.get("revision")
    if not rev:
        issues.append("hub.revision kosong (belum dipublikasikan)")
    elif not _COMMIT_RE.match(rev):
        issues.append(f"hub.revision bukan commit sha 40 hex: {rev}")
    files = []
    for name, e in manifest.get("models", {}).items():
        files.append(e.get("file"))
        if not e.get("file"):
            issues.append(f"{name}: file kosong")
        sha = e.get("sha256")
        if not sha:
            issues.append(f"{name}: sha256 kosong")
        elif not _SHA256_RE.match(sha):
            issues.append(f"{name}: sha256 bukan 64 hex")
        if e.get("size") is None:
            issues.append(f"{name}: size kosong")
    if len(set(files)) != len(files):
        issues.append("nama file di manifest tidak unik")
    for name, e in manifest.get("external", {}).items():
        if not _COMMIT_RE.match(e.get("revision") or ""):
            issues.append(f"external.{name}: revision bukan commit sha 40 hex")
    return issues


def verify_all(offline: bool = False, manifest: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
    """Coba peroleh setiap model; kembalikan {nama: 'ok' | pesan error}. Untuk smoke test."""
    manifest = manifest if manifest is not None else load_manifest()
    result: Dict[str, str] = {}
    for name in manifest.get("models", {}):
        try:
            ensure_model(name, offline=offline, manifest=manifest)
            result[name] = "ok"
        except Exception as e:
            result[name] = f"{type(e).__name__}: {e}".splitlines()[0]
    return result
