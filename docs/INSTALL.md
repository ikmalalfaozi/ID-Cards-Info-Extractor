# Instalasi

Python 3.10-3.13 (3.11 dan 3.13 diuji penuh; Colab saat ini memakai 3.13).

## Ringkasan

`pip install -r requirements.txt` cukup untuk dependensi runtime (CPU). DocAligner tidak lagi dipasang sebagai paket:
modelnya dijalankan langsung lewat onnxruntime (`src/image_alignment/corner_model.py`), jadi tidak ada pin
`capybara`/`docaligner`, `libturbojpeg`, atau `pyheif`. `scripts/install.sh` hanya menambah torch bila belum ada dan
memilih onnxruntime CPU/GPU.

- **numpy dan opencv tidak dikunci** (`numpy>=1.26`, `opencv-python>=4.10`; numpy 1.26 tidak punya wheel untuk Python 3.13).
  Diuji: keluaran sudut di Python 3.11/numpy 1.26/opencv 4.9 identik dengan DocAligner asli; JSON Donut sama di
  Python 3.13/numpy 2.5/opencv 4.14.
- **onnxruntime** `>=1.20,<2` (batas bawah yang punya wheel Python 3.13). Diuji 1.20.1 (Python 3.11) dan 1.31.1
  (Python 3.13, numpy 2.5, opencv 4.14): sudut model heatmap identik; model `point` selisih <= 0,0005 piksel.
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
`bash scripts/install.sh gpu` hanya jika CUDA 12 + cuDNN 9 tersedia. Dengan onnxruntime CPU, `CornerDetector`
memakai CPU dan hasilnya sama.

## Linux / WSL lokal

```bash
python3.11 -m venv .venv && source .venv/bin/activate     # atau: uv venv --python 3.11
bash scripts/install.sh cpu                                # atau gpu
```

## Windows (belum diuji)

```powershell
py -3.11 -m venv .venv ; .venv\Scripts\activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Tidak ada langkah tambahan: semua dependensi berupa wheel PyPI. Belum dijalankan di Windows.

## Device (GPU/CPU)

Pemilihan device dipusatkan di `src/config.py` (`get_device`). Prioritas: argumen `device=` pada kelas
(`DocDetector`, `DocOrientationDetector`, `CornerDetector`, `DonutInfoExtractor`, `create_model`) > variabel lingkungan
`IDCARD_DEVICE` (`auto`, `cpu`, `cuda`, `cuda:N`) > otomatis (CUDA bila ada). Meminta CUDA tanpa GPU tidak menggagalkan
proses: dipakai CPU dengan peringatan. Contoh paksa CPU: `IDCARD_DEVICE=cpu python examples/pipeline_stages.py ...`.

`CornerDetector(model_type="heatmap", model_cfg=None)` menjalankan model DocAligner langsung lewat onnxruntime
(tanpa paket `docaligner`/`capybara`). Varian: heatmap `fastvit_sa24` (bawaan), `fastvit_t8`, `lcnet100`; point `lcnet050`.
Model terdaftar di manifest (`doc-corner-*`) dan diunduh otomatis saat pertama dipakai. Runtime-nya onnxruntime,
bukan torch: GPU dipakai hanya bila `onnxruntime-gpu` terpasang (`requirements/onnx-gpu.txt`); selain itu CPU dan
mencetak pemberitahuan.

Backend torch (opsional): `CornerDetector(backend="torch")` menjalankan model yang sama lewat `onnx2torch`
(`pip install -r requirements/torch-backend.txt`; `onnxruntime` tidak dibutuhkan). Device mengikuti `get_device()`.
Diuji di CPU: sudut model heatmap identik dengan onnxruntime, model `point` selisih <= 0,0012 piksel; torch
lebih lambat (heatmap sekitar 1,2-1,6x, `point` sekitar 12x). GPU dan dataset laporan belum diuji. Bawaannya tetap `onnx`.

onnxruntime dan matplotlib diimpor malas: `import src.image_alignment` (deteksi dokumen, warp, orientasi) berhasil
walaupun keduanya belum terpasang; `CornerDetector()` baru memberi `ImportError` berisi perintah instalasi saat dibuat.
Mode `interactive=True` membutuhkan backend matplotlib interaktif (tidak jalan di Colab/Kaggle).

## Kecepatan Donut (opsional)

`DonutInfoExtractor` memakai setelan yang sama dengan tesis secara bawaan (fp32, beam search 3, `max_length` = batas model).
Opsi berikut mengubah kecepatan dan sebaiknya divalidasi (mis. CER) sebelum dipakai untuk angka yang dilaporkan:

| Opsi | Efek |
|---|---|
| `max_length=300` (konstruktor atau per panggilan) | memotong generasi yang tak berujung; tidak mengubah keluaran yang valid. Tidak boleh melebihi batas model (1536) |
| `num_beams=1` | greedy, lebih cepat (uji CPU: ±1,4x); keluaran bisa berbeda dari beam 3 |
| `predict_batch(images, batch_size=...)` | satu `generate` untuk beberapa gambar; terutama bermanfaat di GPU |
| `dtype="bfloat16"` / `"float16"` | `float16` hanya CUDA; belum divalidasi |
| `quantize="int8"` | kuantisasi dinamis, hanya CPU; ±1,4x lebih cepat tetapi keluaran berubah |

Uji CPU (i5-10210U, 3 gambar ter-align): bawaan baru identik dengan kode lama; `predict_batch`, `num_beams=1`, dan
`max_length=300` menghasilkan JSON yang sama pada 3 gambar itu (sampel kecil, bukan bukti CER). Generasi tak berujung
(mis. foto mentah yang belum di-align) berjalan sampai 1536 token dan puluhan kali lebih lambat.

## Model

Model tidak disimpan di git. Semuanya diunduh otomatis pada pemakaian pertama dan diverifikasi sha256-nya:

| Model | Sumber | Dipin lewat |
|---|---|---|
| YOLO (`doc-seg`, `doc-oc`, `doc-type-cls`), bobot NAFNet/NAFSSR (7), dan DocAligner (`doc-corner-*`, 4 ONNX) | repo Hugging Face `ikmalalfaozi/id-cards-extractor-models` (publik, tag `v1.1`) | commit sha + sha256 di `src/model_manifest.json` |
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
| `requirements.txt` | `base.txt` + `onnx-cpu.txt` (pemasangan satu perintah, CPU) |
| `requirements/base.txt` | dependensi inti (tanpa torch) |
| `requirements/onnx-cpu.txt` / `onnx-gpu.txt` | ONNX Runtime (pilih satu) |
| `requirements/torch-backend.txt` | opsional: `onnx2torch` untuk `CornerDetector(backend="torch")` |
| `requirements/demo.txt` | streamlit dan matplotlib (demo web, mode interaktif) |
| `requirements/dev.txt` | pytest |
