#!/usr/bin/env python
"""Unggah model ke Hugging Face Hub dan perbarui src/model_manifest.json (lihat docs/model-storage-design.md).

Default HANYA rencana (dry-run): tidak mengunduh, tidak mengunggah, tidak mengubah manifest.

  python scripts/upload_models.py                       # lihat rencana + status berkas lokal
  python scripts/upload_models.py --download            # unduh bobot NAFNet yang belum ada (gdown), lalu rencana
  python scripts/upload_models.py --only nafnet-sidd-w32 --download    # ulangi satu model (mis. kuota Drive)
  huggingface-cli login
  python scripts/upload_models.py --execute --tag v1.0 --license other --yolo-license <lisensi-yolo>
  python scripts/upload_models.py --execute --tag v1.0 --license <id-lisensi-tunggal>   # mis. satu lisensi untuk semua

--execute membuat repo publik (bila belum ada), mengunggah semua berkas + README.md + LICENSES/ dalam satu
commit, membuat tag, memeriksa sha256 sisi server, lalu menulis ulang manifest. Tag yang sudah ada ditolak
(tag di HF dapat dipindahkan; rilis baru harus memakai tag baru).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import model_store as ms  # noqa: E402

NAFNET_LICENSE_URL = "https://raw.githubusercontent.com/megvii-research/NAFNet/main/LICENSE"
NAFNET_REPO_URL = "https://github.com/megvii-research/NAFNet"
DEFAULT_WORKDIR = Path.home() / ".cache" / "idcard_extractor" / "upload"


def fmt_size(n: Optional[int]) -> str:
    return "-" if n is None else f"{n / 1024 / 1024:.1f} MiB"


# ------------------------------------------------------------------ pengumpulan
def find_local(entry: Dict[str, Any], yolo_dir: Path, workdir: Path) -> Optional[Path]:
    name = Path(entry["file"]).name
    for base in (yolo_dir, workdir):
        p = base / name
        if p.is_file():
            return p
    return None


def download_from_drive(entry: Dict[str, Any], workdir: Path) -> Optional[Path]:
    gid = (entry.get("fallback") or {}).get("gdrive")
    if not gid:
        return None
    import gdown

    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / Path(entry["file"]).name
    try:
        out = gdown.download(id=gid, output=str(dest), quiet=False)
    except Exception as e:
        print(f"  ! gdown gagal untuk {dest.name}: {type(e).__name__}: {e}")
        out = None
    if not out or not dest.is_file():
        dest.unlink(missing_ok=True)  # jangan tinggalkan berkas parsial
        return None
    return dest


def gather(manifest: Dict[str, Any], yolo_dir: Path, workdir: Path, only: Optional[List[str]] = None,
           download: bool = False) -> List[Dict[str, Any]]:
    models = manifest["models"]
    keys = only if only else list(models)
    unknown = [k for k in keys if k not in models]
    if unknown:
        raise SystemExit(f"Kunci tidak ada di manifest: {', '.join(unknown)}. Tersedia: {', '.join(models)}")
    rows = []
    for key in keys:
        entry = models[key]
        path = find_local(entry, yolo_dir, workdir)
        if path is None and download:
            print(f"- mengunduh {key} dari Google Drive ...")
            path = download_from_drive(entry, workdir)
        row = {"key": key, "file": entry["file"], "path": path, "manifest_sha": entry.get("sha256"),
               "sha256": None, "size": None, "status": "missing"}
        if path is not None:
            row["sha256"] = ms.sha256_file(path)
            row["size"] = path.stat().st_size
            same = row["manifest_sha"] in (None, row["sha256"])
            row["status"] = "ok" if same else "changed"
        rows.append(row)
    return rows


# --------------------------------------------------------------------- konten
def fetch_license_text(opener: Callable[..., Any] = urllib.request.urlopen) -> str:
    with opener(NAFNET_LICENSE_URL, timeout=30) as r:
        return r.read().decode("utf-8")


LICENSE_NAME_OTHER = "per-file-license"
LICENSE_LINK_OTHER = "LICENSE.md"


def build_license_md(rows: List[Dict[str, Any]], yolo_license: str, yolo_license_link: Optional[str] = None) -> str:
    """Tabel lisensi per kelompok berkas (dipakai bila license: other)."""
    yolo = yolo_license + (f" ([teks lisensi]({yolo_license_link}))" if yolo_license_link else "")
    lines = ["# Lisensi per berkas", "",
             "Repo ini berisi berkas dengan lisensi berbeda. Lisensi tiap kelompok berkas:", "",
             "| Berkas | Lisensi | Keterangan |", "|---|---|---|"]
    if any(r["file"].startswith("yolo/") for r in rows):
        lines.append(f"| `yolo/*.pt` | {yolo} | Model YOLO (Ultralytics) milik pemilik repo. |")
    if any(r["file"].startswith("nafnet/") for r in rows):
        lines.append("| `nafnet/*.pth` | MIT (kode NAFNet, megvii-model 2022; bagian BasicSR Apache-2.0) | "
                     f"Bobot resmi dari penulis ([megvii-research/NAFNet]({NAFNET_REPO_URL})), tidak dimodifikasi. "
                     "Penulis tidak menyebut lisensi bobot secara terpisah. Teks: "
                     "`LICENSES/NAFNet-and-BasicSR-LICENSE.txt`. |")
    return "\n".join(lines) + "\n"


def build_model_card(rows: List[Dict[str, Any]], repo_id: str, tag: str, license_id: str) -> str:
    if license_id == "other":
        front = f"license: other\nlicense_name: {LICENSE_NAME_OTHER}\nlicense_link: {LICENSE_LINK_OTHER}"
        license_section = "\n## Lisensi\n\nLisensi berbeda per kelompok berkas; lihat [LICENSE.md](LICENSE.md).\n"
    else:
        front = f"license: {license_id}"
        license_section = ""
    table = "\n".join(f"| `{r['key']}` | `{r['file']}` | {fmt_size(r['size'])} | `{r['sha256']}` |" for r in rows)
    return f"""---
{front}
tags:
- yolo
- nafnet
- document-detection
- ocr
- indonesia
---

# ID Cards Info Extractor: model

Kumpulan model untuk pipeline ekstraksi informasi KTP/SIM/Paspor
([ID-Cards-Info-Extractor]({'https://github.com/ikmalalfaozi/ID-Cards-Info-Extractor'})).
Rilis: `{tag}`. Repo ini dipakai oleh `src/model_store.py` dengan commit sha dan sha256 yang dicatat di
`src/model_manifest.json`; sebaiknya jangan mengunduh berkas tanpa memeriksa sha256.

{license_section}
## Isi

| Kunci | Berkas | Ukuran | sha256 |
|---|---|---|---|
{table}

## Asal

- `yolo/doc-seg.pt`: segmentasi dokumen. `yolo/doc-oc.pt`: klasifikasi orientasi dokumen (0/90/180/270).
  `yolo/doc-type-cls.pt`: klasifikasi jenis dokumen (KTP, SIM, Passport, Other). Model YOLO (Ultralytics).
- `nafnet/*.pth`: bobot resmi **NAFNet / NAFSSR** dari penulis aslinya ([megvii-research/NAFNet]({NAFNET_REPO_URL})),
  tidak dimodifikasi dan disalin ulang di sini agar dapat diunduh tanpa Google Drive. Kode NAFNet berlisensi MIT
  (megvii-model, 2022) dengan bagian BasicSR berlisensi Apache-2.0; teks lisensinya ada di
  `LICENSES/NAFNet-and-BasicSR-LICENSE.txt`. Penulis tidak menyebut lisensi bobot secara terpisah.
  Rujukan: Chen dkk., "Simple Baselines for Image Restoration", ECCV 2022.
- Model OCR Donut tidak ada di repo ini (`ikmalalfaozi/donut-base-finetuned-ktp-sim-passport-v3`).

## Pemakaian

```python
from src.model_store import ensure_model   # dari repo GitHub di atas
path = ensure_model("doc-seg")             # mengunduh dari {repo_id} pada revisi yang dipin, memverifikasi sha256
```
"""


def build_operations(rows: List[Dict[str, Any]], readme: str, license_text: str,
                     license_md: Optional[str] = None) -> list:
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=r["file"], path_or_fileobj=str(r["path"])) for r in rows]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode("utf-8")))
    ops.append(CommitOperationAdd(path_in_repo="LICENSES/NAFNet-and-BasicSR-LICENSE.txt",
                                  path_or_fileobj=license_text.encode("utf-8")))
    if license_md:
        ops.append(CommitOperationAdd(path_in_repo=LICENSE_LINK_OTHER, path_or_fileobj=license_md.encode("utf-8")))
    return ops


# ------------------------------------------------------------------- publikasi
def publish(api: Any, repo_id: str, tag: str, ops: list) -> str:
    """Buat repo (publik) bila belum ada, satu commit, lalu tag. Kembalikan commit sha."""
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    existing = {t.name for t in api.list_repo_refs(repo_id=repo_id, repo_type="model").tags}
    if tag in existing:
        raise SystemExit(f"Tag '{tag}' sudah ada di {repo_id}. Gunakan tag baru (mis. v1.1); tag lama tidak dipindahkan.")
    commit = api.create_commit(repo_id=repo_id, repo_type="model", operations=ops,
                               commit_message=f"Release {tag}")
    api.create_tag(repo_id=repo_id, repo_type="model", tag=tag, revision=commit.oid,
                   tag_message=f"Release {tag}")
    return commit.oid


def verify_remote(api: Any, repo_id: str, oid: str, rows: List[Dict[str, Any]]) -> List[str]:
    """Cocokkan sha256 lokal dengan sha256 LFS di server pada revisi `oid`. Kembalikan daftar masalah."""
    infos = api.get_paths_info(repo_id=repo_id, paths=[r["file"] for r in rows], revision=oid, repo_type="model")
    remote = {i.path: (getattr(i, "lfs", None) or {}).get("sha256") for i in infos}
    problems = []
    for r in rows:
        got = remote.get(r["file"])
        if got is None:
            problems.append(f"{r['file']}: tidak ada di server atau bukan berkas LFS (tidak bisa diverifikasi)")
        elif got != r["sha256"]:
            problems.append(f"{r['file']}: sha256 server {got} != lokal {r['sha256']}")
    return problems


def update_manifest(manifest: Dict[str, Any], rows: List[Dict[str, Any]], repo_id: str, tag: str,
                    oid: str) -> Dict[str, Any]:
    new = copy.deepcopy(manifest)
    new["hub"].update({"repo_id": repo_id, "tag": tag, "revision": oid})
    for r in rows:
        new["models"][r["key"]].update({"sha256": r["sha256"], "size": r["size"]})
    return new


def print_plan(rows: List[Dict[str, Any]], repo_id: str, tag: str) -> None:
    print(f"\nRepo: {repo_id} (publik)   Tag: {tag}\n")
    print(f"{'kunci':18s} {'berkas':38s} {'ukuran':>10s}  status")
    for r in rows:
        print(f"{r['key']:18s} {r['file']:38s} {fmt_size(r['size']):>10s}  {r['status']}")
    print()


# ------------------------------------------------------------------------ main
def main(argv: Optional[List[str]] = None, api: Any = None, license_fetcher: Callable[[], str] = fetch_license_text,
         confirm: Callable[[str], str] = input) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", help="default: hub.repo_id di manifest")
    ap.add_argument("--tag", help="default: hub.tag di manifest")
    ap.add_argument("--license", dest="license_id", help="id lisensi untuk model card (wajib dengan --execute)")
    ap.add_argument("--yolo-license", help="lisensi bobot YOLO untuk tabel per berkas (wajib bila --license other)")
    ap.add_argument("--yolo-license-link", help="URL teks lisensi YOLO (opsional)")
    ap.add_argument("--yolo-dir", type=Path, default=ROOT / "models", help="folder berkas YOLO lokal")
    ap.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR, help="folder unduhan NAFNet (di luar /tmp)")
    ap.add_argument("--only", nargs="+", metavar="KUNCI", help="batasi ke kunci manifest tertentu")
    ap.add_argument("--download", action="store_true", help="unduh berkas yang belum ada dari Google Drive")
    ap.add_argument("--accept-changes", action="store_true",
                    help="terima berkas lokal yang sha256-nya berbeda dari manifest")
    ap.add_argument("--execute", action="store_true", help="benar-benar mengunggah (default: hanya rencana)")
    ap.add_argument("--yes", action="store_true", help="lewati konfirmasi interaktif")
    args = ap.parse_args(argv)

    manifest = ms.load_manifest()
    repo_id = args.repo_id or manifest["hub"]["repo_id"]
    tag = args.tag or manifest["hub"]["tag"]
    rows = gather(manifest, args.yolo_dir, args.workdir, args.only, args.download)
    print_plan(rows, repo_id, tag)

    missing = [r["key"] for r in rows if r["status"] == "missing"]
    changed = [r["key"] for r in rows if r["status"] == "changed"]
    if changed and not args.accept_changes:
        print("Berkas lokal berbeda dari sha256 di manifest: " + ", ".join(changed)
              + "\nPeriksa apakah itu disengaja; bila ya, jalankan dengan --accept-changes.")
        return 2

    if not args.execute:
        if missing:
            print("Belum ada: " + ", ".join(missing) + "  (gunakan --download untuk mengunduh NAFNet dari Drive)")
        print("Dry-run: tidak ada yang diunggah. Tambahkan --execute --license <id> untuk mengunggah.")
        return 0

    if missing:
        print("Tidak dapat mengunggah, berkas belum ada: " + ", ".join(missing) + "\nGunakan --download / --only.")
        return 2
    if not args.license_id:
        print("--license wajib untuk --execute (id tunggal, atau 'other' dengan --yolo-license; lihat docs bagian 10).")
        return 2
    if args.license_id == "other" and not args.yolo_license:
        print("--license other membutuhkan --yolo-license (lisensi bobot YOLO milik Anda); tidak ada nilai default.")
        return 2

    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as e:
        print(f"Belum login ke Hugging Face ({type(e).__name__}). Jalankan: huggingface-cli login")
        return 2
    print(f"Login sebagai: {who}")
    if not args.yes and confirm(f"Unggah {len(rows)} berkas ke {repo_id} (PUBLIK) dengan tag {tag}? Ketik 'unggah': ") != "unggah":
        print("Dibatalkan.")
        return 1

    readme = build_model_card(rows, repo_id, tag, args.license_id)
    license_md = build_license_md(rows, args.yolo_license, args.yolo_license_link) if args.license_id == "other" else None
    oid = publish(api, repo_id, tag, build_operations(rows, readme, license_fetcher(), license_md))
    print(f"Commit: {oid}")

    problems = verify_remote(api, repo_id, oid, rows)
    if problems:
        print("Verifikasi sisi server GAGAL, manifest TIDAK diubah:\n  - " + "\n  - ".join(problems))
        return 3

    new = update_manifest(manifest, rows, repo_id, tag, oid)
    with open(ms.MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2)
        f.write("\n")
    issues = ms.manifest_issues(new)
    print("Manifest diperbarui: " + str(ms.MANIFEST_PATH))
    print("Manifest siap rilis." if not issues else "Masih ada yang kosong:\n  - " + "\n  - ".join(issues))
    print("Selanjutnya: git diff src/model_manifest.json, lalu commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
