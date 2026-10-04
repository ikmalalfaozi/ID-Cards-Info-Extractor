# Instalasi

Python 3.10-3.12. Python 3.13+ belum didukung (paket yang dikunci belum tentu punya wheel).

## Kenapa tidak cukup `pip install -r requirements.txt`

`capybara-docsaid==0.6.0` (dibutuhkan DocAligner) mengunci `onnxruntime_gpu==1.20.1` di Linux, versi yang
tidak ada di PyPI, sehingga pip gagal me-resolve. Karena itu `capybara-docsaid` dan `docaligner-docsaid`
dipasang dengan `--no-deps`, dan dependensinya dicantumkan manual. `scripts/install.sh` melakukan urutan ini.

Hal lain yang perlu diketahui:
- **numpy < 2 dan opencv-python 4.9.0.80** dipaksa oleh capybara. Di Colab/Kaggle numpy akan diturunkan,
  jadi **runtime/kernel harus di-restart** setelah instalasi.
- **`libturbojpeg`** (library sistem) wajib ada saat `import capybara`. Skrip memasangnya lewat apt.
- **`PyTurboJPEG<2`**: versi 2.x butuh libjpeg-turbo 3, sedangkan Ubuntu/Colab/Kaggle membawa 2.x.
- torch/torchvision tidak ada di `requirements/base.txt` agar torch bawaan Colab/Kaggle tidak ditimpa.
  Versi yang diuji: torch 2.6.0 (CPU), Python 3.11.

## Colab / Kaggle

Aktifkan GPU (opsional), lalu jalankan di sel pertama:

```bash
!git clone <URL-repo> ID-Cards-Info-Extractor
%cd ID-Cards-Info-Extractor
!bash scripts/install.sh auto
```

Lalu **restart runtime** (Colab: Runtime > Restart session; Kaggle: Run > Restart & clear cell outputs)
dan jalankan sel uji:

```python
import sys; sys.path.insert(0, ".")          # atau %cd ke root repo
!python examples/pipeline_stages.py "demo_input/doni.jpg" --until preprocess
```

Untuk OCR (Donut, unduhan sekitar 777 MB): hapus `--until preprocess`.

Jika inferensi DocAligner di GPU gagal memuat CUDA provider (kesalahan cuDNN/CUDA), pasang varian CPU:

```bash
!pip uninstall -y onnxruntime-gpu && pip install -r requirements/onnx-cpu.txt
```

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

## Struktur requirements

| File | Isi |
|---|---|
| `requirements/base.txt` | dependensi inti (tanpa torch) |
| `requirements/onnx-cpu.txt` / `onnx-gpu.txt` | ONNX Runtime (pilih satu) |
| `requirements/docaligner-deps.txt` | dependensi capybara yang dipasang manual |
| `requirements/docaligner.txt` | capybara + docaligner, pasang dengan `--no-deps` |
| `requirements/demo.txt` | streamlit (hanya untuk demo) |
