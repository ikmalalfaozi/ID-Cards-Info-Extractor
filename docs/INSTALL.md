# Instalasi

Python 3.10-3.13 (3.11 dan 3.13 diuji penuh; Colab saat ini memakai 3.13).

## Kenapa tidak cukup `pip install -r requirements.txt`

`capybara-docsaid==0.6.0` (dibutuhkan DocAligner) mengunci `onnxruntime_gpu==1.20.1` di Linux, versi yang
tidak ada di PyPI, sehingga pip gagal me-resolve. Karena itu `capybara-docsaid` dan `docaligner-docsaid`
dipasang dengan `--no-deps`, dan dependensinya dicantumkan manual. `scripts/install.sh` melakukan urutan ini.

Hal lain yang perlu diketahui:
- **numpy dan opencv tidak dikunci** (`numpy>=1.26`, `opencv-python>=4.10`). Metadata capybara meminta
  `numpy<2` dan `opencv==4.9.0.80`, tetapi itu hanya metadata (capybara dipasang `--no-deps`) dan numpy 1.26
  tidak punya wheel untuk Python 3.13. Diuji: keluaran DocAligner di Python 3.11/numpy 1.26/opencv 4.9 dan
  Python 3.13/numpy 2.5/opencv 4.14 sama (selisih sudut <= 0,05 piksel), begitu pula JSON Donut.
- **pyheif** (Linux, di-import capybara tanpa pengaman) tidak punya wheel untuk Python 3.13. `install.sh`
  mencoba `pip install pyheif`, lalu `apt install libheif-dev`, dan terakhir memasang stub (file `.heic/.heif`
  tidak akan terbaca, JPG/PNG tidak terpengaruh).
- **`libturbojpeg`** (library sistem) wajib ada saat `import capybara`. Skrip memasangnya lewat apt.
- **`PyTurboJPEG<2`**: versi 2.x butuh libjpeg-turbo 3, sedangkan Ubuntu/Colab/Kaggle membawa 2.x.
- torch/torchvision tidak ada di `requirements/base.txt` agar torch bawaan Colab/Kaggle tidak ditimpa.
  Yang diuji: torch 2.6.0 (CPU) di Python 3.11 dan 3.13. Torch bawaan Colab/Kaggle yang lebih baru belum diuji
  (jalur GPU juga belum dijalankan, hanya di-resolve).

## Colab / Kaggle

Aktifkan GPU (opsional), lalu jalankan di sel pertama:

```bash
!git clone <URL-repo> ID-Cards-Info-Extractor
%cd ID-Cards-Info-Extractor
!bash scripts/install.sh auto
```

Jika instalasi mengganti paket yang sudah dimuat kernel (misalnya numpy), **restart runtime** (Colab: Runtime > Restart
session; Kaggle: Run > Restart & clear cell outputs). Lalu jalankan sel uji:

```python
import sys; sys.path.insert(0, ".")          # atau %cd ke root repo
!python examples/pipeline_stages.py "demo_input/doni.jpg" --until preprocess
```

Untuk OCR (Donut, unduhan sekitar 777 MB): hapus `--until preprocess`. Model diunduh dari Hugging Face (lihat bagian Model).

`auto` memasang ONNX Runtime **CPU** walau ada GPU (YOLO dan Donut tetap memakai GPU lewat torch). Model DocAligner
kecil, dan `onnxruntime-gpu` 1.20 butuh CUDA 12 (`libcublas.so.12`): di Colab uji, paket itu gagal memuat CUDA provider
(`Failed to load library libonnxruntime_providers_cuda.so`) lalu jatuh ke CPU dengan pesan error. Pakai
`bash scripts/install.sh gpu` hanya jika CUDA 12 + cuDNN 9 tersedia. Di CPU-only ORT, `Backend.cuda` hanya
memberi peringatan dan hasilnya sama (diuji).

## Linux / WSL lokal

```bash
python3.11 -m venv .venv && source .venv/bin/activate     # atau: uv venv --python 3.11
bash scripts/install.sh cpu                                # atau gpu
```

Tanpa sudo, `libturbojpeg` dapat diekstrak ke folder sendiri lalu ditunjuk lewat `LD_LIBRARY_PATH`:

```bash
apt download libturbojpeg0 && dpkg -x libturbojpeg0_*.deb .venv/native
export LD_LIBRARY_PATH="$PWD/.venv/native/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
```

## Windows (belum diuji)

```powershell
py -3.11 -m venv .venv ; .venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/base.txt -r requirements/onnx-cpu.txt
pip install -r requirements/docaligner-deps.txt
pip install --no-deps -r requirements/docaligner.txt
```

`libjpeg-turbo` untuk Windows harus dipasang terpisah (installer resmi libjpeg-turbo) dan perlu diverifikasi
`import capybara` berhasil. `pyheif` hanya dibutuhkan di Linux (sudah diberi marker di `docaligner-deps.txt`).

## Model

Model tidak disimpan di git. Semuanya diunduh otomatis pada pemakaian pertama dan diverifikasi sha256-nya:

| Model | Sumber | Dipin lewat |
|---|---|---|
| YOLO (`doc-seg`, `doc-oc`, `doc-type-cls`) dan bobot NAFNet/NAFSSR (7) | repo Hugging Face `ikmalalfaozi/id-cards-extractor-models` (publik, tag `v1.0`) | commit sha + sha256 di `src/model_manifest.json` |
| Donut (OCR, ~777 MB) | repo Hugging Face `ikmalalfaozi/donut-base-finetuned-ktp-sim-passport-v3` | commit sha di `src/model_manifest.json` |

Unduhan disimpan di cache Hugging Face (`~/.cache/huggingface`, atau `HF_HOME`), sehingga berkas yang tidak berubah antar rilis
tidak diunduh ulang. Folder `models/` di repo tidak lagi dipakai.

Variabel lingkungan:

| Variabel | Fungsi |
|---|---|
| `HF_HUB_OFFLINE=1` | hanya memakai cache (tanpa jaringan); error jelas bila model belum ada |
| `IDCARD_MODELS_DIR=<folder>` | mirror lokal dengan struktur `yolo/`, `nafnet/` seperti repo Hugging Face |
| `IDCARD_NO_FALLBACK=1` | matikan cadangan `gdown` (Google Drive); dipakai untuk membuktikan unduhan hanya dari Hugging Face |
| `IDCARD_CACHE_DIR` | lokasi cache cadangan `gdown` (default `~/.cache/idcard_extractor`) |

Memakai model sendiri tanpa mengubah manifest: `DocDetector(model_path="...")`, `DocOrientationDetector(model_path="...")`,
`load_doc_type_model("...")`, atau `pretrain_network_g: <path>` di YAML NAFNet (berkas harus sudah ada).

Menyiapkan semua model sebelum offline, dan memeriksa integritasnya:

```python
from src.model_store import verify_all
print(verify_all())                 # {'doc-seg': 'ok', ...}; jalankan dengan internet sekali
```

Merilis model baru (pemilik repo): letakkan berkas di `models/` (YOLO) atau biarkan skrip mengunduh bobot NAFNet,
lalu `python scripts/upload_models.py` (rencana), dan `--execute --tag vX.Y --license other --yolo-license <lisensi>`
untuk mengunggah dan memperbarui manifest. Tag yang sudah ada tidak dipindahkan; rilis baru memakai tag baru.
Lihat `docs/model-storage-design.md`.

## Struktur requirements

| File | Isi |
|---|---|
| `requirements/base.txt` | dependensi inti (tanpa torch) |
| `requirements/onnx-cpu.txt` / `onnx-gpu.txt` | ONNX Runtime (pilih satu) |
| `requirements/docaligner-deps.txt` | dependensi capybara yang dipasang manual |
| `requirements/docaligner.txt` | capybara + docaligner, pasang dengan `--no-deps` |
| `requirements/demo.txt` | streamlit (hanya untuk demo) |
