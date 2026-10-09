# Desain: Penyimpanan dan Pemuatan Model

Status: usulan, belum diimplementasikan. Menggantikan `gdown` (Google Drive) dan model `.pt` di git sebagai jalur utama.

## 1. Keputusan yang sudah diambil
- Repo kode dan repo model **publik**; model boleh dibagikan.
- **Satu repo Hugging Face** untuk semua model buatan sendiri + bobot NAFNet, dengan **tag versi** (`v1.0`, ...).
- Donut tetap di repo HF-nya sendiri (`ikmalalfaozi/donut-base-finetuned-ktp-sim-passport-v3`), di-pin ke commit hash.
- Riwayat git **tidak ditulis ulang**; file `models/*.pt` hanya dikeluarkan dari working tree lewat commit biasa.
- Ketujuh varian NAFNet dipertahankan.

## 2. Inventaris dan penempatan

| Kunci manifest | File di repo HF | Asal | Pemakai di kode |
|---|---|---|---|
| `doc-seg` | `yolo/doc-seg.pt` (6 MB) | git `models/` | `DocDetector` |
| `doc-oc` | `yolo/doc-oc.pt` (3 MB) | git `models/` | `DocOrientationDetector` |
| `doc-type-cls` | `yolo/doc-type-cls.pt` (21 MB) | git `models/` | filter dokumen (contoh membaca langsung) |
| `nafnet-gopro-w32` / `-w64` | `nafnet/NAFNet-GoPro-width32.pth`, `...-width64.pth` | Drive penulis NAFNet | `ImageRestorationModel` |
| `nafnet-reds-w64` | `nafnet/NAFNet-REDS-width64.pth` | idem | idem |
| `nafnet-sidd-w32` / `-w64` | `nafnet/NAFNet-SIDD-width32.pth`, `...-width64.pth` | idem | idem |
| `nafssr-l-2x` / `-4x` | `nafnet/NAFSSR-L_2x.pth`, `NAFSSR-L_4x.pth` | idem | idem |
| `donut` (eksternal) | repo terpisah, commit `ab6bf5f7fef0f4fe6e4d5de505d0b3987104b599` | HF | `DonutInfoExtractor` |

Struktur repo HF `ikmalalfaozi/id-cards-extractor-models`:
```
README.md                      # model card: sumber, lisensi, tabel checksum, cara pakai
LICENSES/NAFNet-MIT.txt
LICENSES/BasicSR-Apache-2.0.txt
yolo/*.pt
nafnet/*.pth
```
DocAligner (`.onnx`) tidak dikelola di sini: diunduh oleh paket `docaligner` sendiri saat pertama dipakai, jadi perlu internet pada penggunaan pertama.

## 3. Manifest (`src/model_manifest.json`, dilacak di git)

Sumber kebenaran tunggal untuk versi. Diperbarui oleh skrip unggah, di-review lewat git.

```json
{
  "schema": 1,
  "hub": {
    "repo_id": "ikmalalfaozi/id-cards-extractor-models",
    "tag": "v1.0",
    "revision": "<commit sha 40 hex dari tag v1.0>"
  },
  "models": {
    "doc-seg": {
      "file": "yolo/doc-seg.pt",
      "sha256": "<64 hex>", "size": 5979549,
      "fallback": {"gdrive": "1Ny156vx7sc6ux_1Ay-gJMYSZwuK6eywn"}
    },
    "nafnet-gopro-w32": {
      "file": "nafnet/NAFNet-GoPro-width32.pth",
      "sha256": "<64 hex>", "size": 72000000,
      "fallback": {"gdrive": "1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj"}
    }
  },
  "external": {
    "donut": {
      "repo_id": "ikmalalfaozi/donut-base-finetuned-ktp-sim-passport-v3",
      "revision": "ab6bf5f7fef0f4fe6e4d5de505d0b3987104b599"
    }
  }
}
```

Aturan:
- Kode **memakai `revision` (commit sha)**, bukan `tag`: tag di HF bisa dipindahkan, sha tidak. `tag` hanya penanda untuk manusia.
- `sha256` memverifikasi isi file setelah diunduh. Nilai `size` dan `sha256` untuk `doc-*` sudah dapat diisi dari berkas lokal; ID Drive diambil dari kode/YAML yang ada. Nilai lain diisi skrip unggah.
- Pin ganda (revision + sha256) juga melindungi dari repo HF yang terkompromi: file `.pt`/`.pth` berformat pickle, jadi file yang diubah tidak boleh dimuat.

## 4. Komponen: `src/model_store.py`

```python
def ensure_model(name: str, *, offline: bool | None = None, verify: bool = True) -> Path:
    """Path lokal untuk model `name` dari manifest; mengunduh bila perlu."""

def external_revision(name: str) -> tuple[str, str]:   # (repo_id, revision), mis. untuk Donut
def list_models() -> dict[str, dict]:
def verify_all(offline: bool = False) -> dict[str, bool]:   # untuk smoke test / CI
```

Alur `ensure_model(name)`:

```
manifest ---> entri `name`
   |
   1. env IDCARD_MODELS_DIR set dan <dir>/<file> ada? ---ya---> verifikasi sha256 ---> return
   |  tidak
   2. hf_hub_download(repo_id, file, revision=<sha>, local_files_only=offline)
   |      sukses ---> verifikasi sha256 ---> return
   |      gagal karena jaringan/HTTP ---+
   |      gagal karena offline & tidak ada di cache ---> error "model tidak ada di cache" (tanpa fallback)
   3. fallback gdown (hanya bila ada ID dan offline=False; simpan di ~/.cache/idcard_extractor/<sha256>/)
   |      sukses ---> verifikasi sha256 ---> return
   4. gagal semua ---> ModelUnavailableError (pesan: sumber yang dicoba + penyebab + cara manual)
```

Keputusan desain:
- **Cache memakai cache bawaan Hugging Face** (`~/.cache/huggingface/hub`, atau `HF_HOME`/`HF_HUB_CACHE`), bukan folder buatan sendiri. Resume unduhan, file lock, dan penyimpanan berbasis konten (file yang tidak berubah antar tag tidak diunduh ulang) sudah disediakan `huggingface_hub` 0.36.2 (terpasang).
- **Offline:** `offline=True` atau `HF_HUB_OFFLINE=1` hanya memakai cache.
- **Checksum tidak sama = error keras, bukan fallback**, supaya file rusak/berubah tidak dimuat diam-diam.
- **`IDCARD_MODELS_DIR`** untuk mirror lokal (jaringan tertutup): struktur subfolder sama dengan repo HF.
- `IDCARD_NO_FALLBACK=1` mematikan jalur `gdown` (dipakai saat pengujian agar terbukti tidak ada lalu lintas Drive).
- Tanpa dependensi baru: `huggingface_hub` sudah dibawa `transformers`; `gdown` sudah ada.
- Path ke manifest dihitung dari lokasi file (`Path(__file__).parent`), bukan CWD.

## 5. Perubahan pada kode yang ada

| File | Perubahan |
|---|---|
| `src/image_alignment/doc_detector.py` | `__init__(self, model_path=None)`; `None` -> `ensure_model("doc-seg")`. `model_save_path` tetap diterima sebagai alias (override lokal); `google_drive_file_id` diterima tetapi diabaikan dengan `DeprecationWarning`. Hapus blok unduhan `gdown` |
| `src/image_alignment/orientation.py` | sama, kunci `doc-oc` |
| `src/utils.py` | tambah `load_doc_type_model(model_path=None) -> YOLO` (kunci `doc-type-cls`) |
| `src/nafnet/model.py` | `path.pretrain_model` (kunci manifest) dipakai bila ada; `path.pretrain_network_g` tetap sebagai override lokal eksplisit. Hapus blok `gdown`. (Bug `opt['dist']` dibereskan bersamaan, lihat `docs/portability-design.md` F6) |
| `nafnet-options/*.yaml` (7) | ganti `pretrain_network_g` + `pretrain_network_g_gdrive_id` dengan `pretrain_model: <kunci>` |
| `src/ocr/donut.py` | `DonutInfoExtractor(model_name=None, revision=None)` default dari `external_revision("donut")`; `download_model(..., revision, cache_dir=None)` meneruskan `revision=` ke `from_pretrained`; default cache = cache HF (tidak lagi `./models`) |
| `examples/pipeline_stages.py` | `stage_filter` memakai `load_doc_type_model()` |
| `.gitignore` | `models/` (setelah file `.pt` dikeluarkan dari working tree) |
| `docs/INSTALL.md` | bagian "Model": lokasi cache, variabel lingkungan, mode offline |

Kompatibilitas: pemanggil lama `DocDetector()` / `DocOrientationDetector()` tanpa argumen tetap jalan (sekarang lewat manifest).

## 6. Skrip unggah dan prosedur rilis (`scripts/upload_models.py`)

Dijalankan **oleh pemilik repo** (butuh token tulis HF: `huggingface-cli login`). Default `--dry-run`.

1. Kumpulkan berkas: YOLO dari `models/` (sebelum dikeluarkan dari working tree), NAFNet diunduh dari ID Drive ke folder sementara (`gdown`).
2. Hitung `sha256` dan `size` tiap berkas.
3. Buat repo HF (publik) bila belum ada; `upload_folder` memakai struktur di bagian 2, termasuk `README.md` (model card) dan `LICENSES/`.
4. `create_tag(v1.0)`, ambil commit sha dari tag.
5. Tulis ulang `src/model_manifest.json` (revision, sha256, size) dan cetak diff; pengguna meng-commit manifest di git.
6. Rilis berikutnya (`v1.1`): hanya berkas yang berubah diunggah; manifest memakai sha baru; berkas yang tidak berubah tidak diunduh ulang di klien karena cache berbasis konten.

Model card memuat sumber dan lisensi: NAFNet MIT (megvii-model, 2022) dengan bagian BasicSR Apache-2.0, tautan ke repo asli, dan catatan bahwa lisensi bobot tidak disebut terpisah oleh penulisnya.

## 7. Pengujian

| Uji | Cara |
|---|---|
| Skema manifest | semua kunci punya `file`, `sha256` 64 hex, `revision` 40 hex; nama file unik |
| `ensure_model` jalur HF | `hf_hub_download` di-monkeypatch ke file palsu; hasil path benar dan sha diverifikasi |
| Checksum salah | harus melempar error, tanpa fallback |
| Offline, cache kosong | error yang jelas; tidak ada percobaan jaringan |
| Fallback `gdown` | HF dibuat gagal; `gdown` palsu dipakai; `IDCARD_NO_FALLBACK=1` menonaktifkannya |
| `IDCARD_MODELS_DIR` | folder sementara dengan struktur sama |
| Integrasi (manual) | instalasi bersih lokal + Colab + Kaggle, jalankan `examples/pipeline_stages.py` sampai OCR dengan `IDCARD_NO_FALLBACK=1` |

## 8. Risiko

| Risiko | Mitigasi |
|---|---|
| Unduhan Drive untuk bobot NAFNet bisa kena kuota saat menyiapkan unggahan awal | unduh satu per satu, ulangi nanti; sumber alternatif Baidu dari repo NAFNet bila perlu |
| Ukuran dan kecepatan unduhan di Colab/Kaggle (NAFSSR bisa jauh lebih besar dari 68 MB) | unduh per kebutuhan (lazy), bukan semua sekaligus |
| Tag HF dapat dipindahkan | kode memakai commit sha + sha256 |
| Simlink di Windows tanpa mode developer | `huggingface_hub` otomatis menyalin file dan hanya memberi peringatan; set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| Repo HF tidak tersedia / akun dibekukan | `IDCARD_MODELS_DIR` dan fallback `gdown` selama ID Drive masih ada |
| Klon lama yang punya `models/*.pt` | tidak dipakai lagi; mereka perlu `git pull` dan unduhan baru (sha dicek, jadi tidak ada ketidakcocokan diam-diam) |

## 9. Tahap implementasi

| Tahap | Isi | Siapa | Verifikasi |
|---|---|---|---|
| P0 | buat akun/token HF, putuskan nama repo final | pengguna | `huggingface-cli whoami` |
| P1 | `model_store.py`, skema manifest (checksum `doc-*` terisi), pengujian tanpa jaringan | Claude | tes unit lulus |
| P2 | `scripts/upload_models.py` (+ model card), jalankan dry-run lalu unggah | Claude menulis, **pengguna menjalankan** | repo HF terlihat, manifest terisi, tag `v1.0` |
| P3 | ganti pemanggil (bagian 5), YAML, `Donut` dengan revision | Claude | pipeline lengkap lolos lokal |
| P4 | uji bersih di Colab dan Kaggle dengan `IDCARD_NO_FALLBACK=1` | pengguna melaporkan | tidak ada lalu lintas Drive, hasil sama dengan acuan |
| P5 | `git rm` `models/*.pt`, `.gitignore`, dokumentasi | Claude | `git status` bersih; clone baru berjalan |

P3 tidak boleh dimulai sebelum P2 selesai (manifest harus punya revision dan sha256 nyata).

## 10. Keputusan terbuka
1. **Lisensi bobot YOLO.** Model dilatih dengan Ultralytics, yang berlisensi AGPL-3.0 (atau Enterprise). Perlu diputuskan lisensi yang dicantumkan di model card; saya tidak dapat menentukan implikasi hukumnya.
2. **Nama repo final** (`id-cards-extractor-models` hanya usulan; pemeriksaan tanpa login memberi 401 sehingga belum bisa dipastikan sudah ada atau belum).
3. **Lisensi model Donut**: kartu modelnya saat ini tidak mencantumkan lisensi.
4. **Baidu sebagai sumber cadangan NAFNet**: perlu atau tidak.
