# Desain Refactor: Portabilitas Lintas Platform

Target: Windows, Linux, Colab, Kaggle (Python 3.10-3.12, GPU opsional).
Kendala tetap: DocAligner dipertahankan (`docaligner-docsaid` 1.1.1 + `capybara-docsaid==0.6.0`) agar hasil TA bisa direproduksi.

## 1. Temuan (dasar desain)

| # | Temuan | Lokasi | Dampak |
|---|--------|--------|--------|
| F1 | `capybara-docsaid==0.6.0` mengunci `onnxruntime_gpu==1.20.1` (Linux), versi ini tidak ada di PyPI. Di Windows tidak ada onnxruntime sama sekali | metadata capybara | `pip install -r` gagal di Colab/Kaggle; di Windows DocAligner gagal saat dijalankan |
| F2 | `docaligner-docsaid` 1.1.1 membatasi `capybara<1.0.0` | metadata docaligner | tidak bisa upgrade capybara |
| F3 | `opencv-python==4.9.0.80` dan `numpy<2` dikunci oleh capybara | metadata capybara | pin ketat di requirements adalah akibat, bukan pilihan |
| F4 | `corner_detector.py` mengimpor `docaligner`, `capybara`, `matplotlib` di level modul, dan `image_alignment/__init__.py` mengimpornya | `src/image_alignment/` | satu paket gagal = seluruh `image_alignment` (termasuk `warp_image`, `DocDetector`) tidak bisa diimpor |
| F5 | Import absolut `from src...` | `nafnet/model.py:10`, `nafnet/utils.py:9`, `image_transform.py:3`, `corner_detector.py:8` | hanya jalan jika root repo ada di `sys.path`; rusak jika dipakai sebagai package |
| F6 | `opt['dist']` dibaca di `model_to_device`, tetapi tidak ada di YAML maupun `parse()` | `nafnet/model.py:27` | `KeyError` saat `ImageRestorationModel` dibuat |
| F7 | `print(' load net keys', load_net.keys)` mencetak method, bukan keys; `DistributedDataParallel` dan `torch.cuda.current_device()` ada di jalur inferensi | `nafnet/model.py` | noise, dan kode mati yang rapuh di CPU |
| F8 | `torch.cuda.synchronize()` tanpa pengecekan | `nafnet/archs.py:283` | error di CPU (hanya fungsi benchmark) |
| F9 | Path model relatif (`models/...`) terhadap working directory; unduhan lewat `gdown` | `doc_detector`, `orientation`, YAML NAFNet | gagal jika dijalankan dari direktori lain; Google Drive sering kena kuota |
| F10 | Logika pemilihan device diulang di 5 tempat | `doc_detector`, `orientation`, `corner_detector`, `model.py`, `donut.py` | tidak konsisten (NAFNet bergantung `num_gpu` di YAML, lainnya `cuda.is_available`) |
| F11 | `polygon_interactor` memakai matplotlib interaktif | `image_alignment/` | tidak jalan headless |
| F12 | `streamlit` dan `matplotlib` termasuk dependensi wajib | `requirements.txt` | membebani install di Colab/Kaggle |
| F13 | Tidak ada `pyproject.toml`; tidak ada pemetaan dependensi implisit Donut (`sentencepiece`, `protobuf`) | root | install tidak deterministik |

## 2. Arsitektur target

```
pyproject.toml            # package `idcard_extractor`, extras: [docaligner] [demo] [dev]
requirements/
  base.txt                # dependensi inti, rentang longgar (>=,<)
  docaligner.txt          # capybara 0.6.0 + docaligner + dependensi capybara manual (dipasang --no-deps)
  onnx-gpu.txt            # onnxruntime-gpu==1.20.*
  onnx-cpu.txt            # onnxruntime==1.20.*
  demo.txt                # streamlit, matplotlib
scripts/install.py        # satu perintah: deteksi OS/GPU lalu pasang lapisan yang tepat
notebooks/colab_kaggle_smoke.ipynb
src/                      # (nama paket tetap `src` dulu, rename belakangan)
  config.py               # satu sumber: device, MODELS_DIR, sumber unduhan
  models_hub.py           # ensure_model(name) -> Path (cache + fallback sumber)
  image_alignment/
    corner_detector.py    # adapter DocAligner, import malas
    ...
  nafnet/                 # inferensi saja
```

Prinsip: **inti ringan, DocAligner terisolasi, satu tempat untuk device dan path model.**

## 3. Keputusan per komponen

### 3.1 Lapisan dependensi (F1, F2, F3, F12, F13)

Urutan instalasi (tidak boleh diubah):
1. `pip install -r requirements/base.txt` (jangan menginstal ulang torch/torchvision; di Colab/Kaggle sudah ada).
2. Pilih satu: `onnx-gpu.txt` (Linux dengan GPU) atau `onnx-cpu.txt` (Windows, Mac, CPU).
3. `pip install --no-deps -r requirements/docaligner.txt`.
4. Dependensi capybara dicantumkan manual di `docaligner.txt` (di luar baris `--no-deps`): `dacite psutil requests onnx colored pdf2image ujson pyyaml tqdm pybase64 PyTurboJPEG dill networkx natsort flask shapely piexif matplotlib pyheif; platform_system=="Linux"`.
   `opencv-python==4.9.0.80` dan `numpy<2` ditulis eksplisit di sini, agar tidak bergantung pada metadata capybara.

Aturan versi:
- `base.txt`: `torch>=2.6,<2.8`, `transformers>=4.41.2,<4.50`, `ultralytics>=8.3.76,<9`, `numpy>=1.26,<2`, tanpa `~=`.
- `opencv-python` diganti `opencv-python-headless==4.9.0.80`? Hanya jika DocAligner/capybara tidak membutuhkan fungsi GUI. Diputuskan setelah uji (lihat Risiko R2).
- Tambahkan dependensi implisit Donut: `sentencepiece`, `protobuf`.

### 3.2 Adapter DocAligner (F4, F11)

- `CornerDetector` mengimpor `docaligner` dan `capybara` di dalam `__init__`, bukan di level modul.
- Jika impor gagal, lempar `ImportError` dengan pesan yang menyebut perintah instalasi yang benar.
- `image_alignment/__init__.py` tidak lagi mengimpor `CornerDetector`, `interactive_get_contour` secara langsung; `warp_image`, `order_points`, `DocDetector`, `DocOrientationDetector` tetap bisa diimpor tanpa DocAligner.
- `order_points` dipindah ke `image_transform.py` (atau modul netral) agar `warp_image` tidak bergantung pada `corner_detector` (menghapus ketergantungan melingkar).
- `interactive_get_contour`/`PolygonInteractor`: impor matplotlib di dalam fungsi, dan jika backend non-interaktif, lempar error yang jelas (`interactive=True` hanya untuk lingkungan lokal).
- Backend DocAligner (`Backend.cuda`/`Backend.cpu`) ditentukan dari `config.get_device()`, dan jatuh ke CPU jika `onnxruntime-gpu` tidak terpasang.

### 3.3 NAFNet (F6, F7, F8)

- `BaseModel.model_to_device`: hapus DDP/DataParallel (hanya inferensi). Cukup `net.to(device)`.
- Device ditentukan oleh `config.get_device()`; `num_gpu` di YAML tidak lagi menentukan device.
- `parse()` tidak perlu menambah `dist`; field itu dihapus dari semua pembacaan.
- Perbaiki `print(... load_net.keys)` menjadi log yang benar atau hapus.
- `torch.load(..., weights_only=True)` (checkpoint hanya berisi tensor; berlaku default di torch 2.6, dibuat eksplisit).
- `measure_inference_speed`: bungkus `torch.cuda.synchronize()` dengan `if torch.cuda.is_available()`.
- Impor relatif (`from .archs import ...`) menggantikan `from src.nafnet...`.
- YAML di `nafnet-options/` hanya menyimpan nama model; path bobot dan sumber unduhan dikelola `models_hub` (3.4).

### 3.4 Path dan unduhan model (F9)

`config.py`:
- `MODELS_DIR` = `$IDCARD_MODELS_DIR` jika diset, selain itu `<root repo>/models` (dihitung dari `__file__`, bukan CWD).
- `get_device()` = `"cuda"` jika `torch.cuda.is_available()`, selain itu `"cpu"`; bisa di-override dengan `$IDCARD_DEVICE`.

`models_hub.ensure_model(name)`:
1. Cari di `MODELS_DIR`; jika ada, pakai.
2. Jika tidak, unduh dari sumber utama (Hugging Face Hub atau GitHub Release, **menunggu keputusan**), dengan fallback ke `gdown` yang sekarang.
3. Verifikasi ukuran/checksum; hapus file parsial jika gagal.
Model yang sudah di-commit (`doc-*.pt`) tetap dipakai apa adanya.

### 3.5 Device tunggal (F10)

Semua kelas (`DocDetector`, `DocOrientationDetector`, `CornerDetector`, `ImageRestorationModel`, `DonutInfoExtractor`) memanggil `config.get_device()` dan menerima parameter `device=None` opsional.

## 4. Urutan implementasi (usulan tahapan)

| Tahap | Isi | Verifikasi |
|-------|-----|------------|
| 1 | `requirements/` + `scripts/install.py` + install DocAligner `--no-deps` | `pip install` bersih di Kaggle dan Colab, `import docaligner` berhasil |
| 2 | `config.py`, `models_hub.py`, ubah path dan device di 5 kelas | jalankan dari direktori lain, tanpa GPU |
| 3 | Adapter DocAligner (import malas, pisah `order_points`) | `import src.image_alignment` tanpa docaligner terpasang tetap jalan |
| 4 | Pembersihan NAFNet (F6-F8, impor relatif) | `ImageRestorationModel` terbentuk di CPU, 1 gambar diproses |
| 5 | `pyproject.toml`, smoke test, notebook Colab/Kaggle | smoke test lulus di 4 lingkungan |

## 5. Risiko dan hal yang perlu diuji

- **R1** `onnxruntime-gpu` 1.20.0/1.20.2 sebagai pengganti 1.20.1: bandingkan keluaran sudut DocAligner pada `demo_input/` (toleransi piksel kecil).
- **R2** `opencv-python-headless` vs `opencv-python`: capybara mungkin memanggil fungsi GUI; diuji sebelum diganti.
- **R3** `onnxruntime-gpu` butuh CUDA 12 + cuDNN 9; sesuai torch 2.6 cu124 di Kaggle, perlu dipastikan di Colab.
- **R4** Python 3.13+ kemungkinan tidak punya wheel untuk paket yang dikunci; target dibatasi 3.10-3.12.
- **R5** Pin `numpy<2` bertahan selama DocAligner dipertahankan; ini batas atas lingkungan yang bisa didukung.

- **R6 (terbukti di uji CPU)** `capybara` membuat `TurboJPEG()` saat diimpor, sehingga `libturbojpeg` (library sistem) wajib ada. `PyTurboJPEG` 2.x butuh libjpeg-turbo 3.x, jadi dipin `PyTurboJPEG<2` agar cocok dengan libjpeg-turbo 2.x (Ubuntu/Colab/Kaggle). Di Linux tanpa sudo, library bisa diekstrak dan dicari lewat `LD_LIBRARY_PATH`. Windows belum diuji.

## 6. Keputusan yang masih terbuka

1. Sumber unduhan model: Hugging Face Hub atau GitHub Release (atau tetap `gdown` + fallback)?
2. Apakah nama paket `src` boleh diganti (mis. `idcard_extractor`) agar bisa di-`pip install`, atau dibiarkan dulu?
3. Varian NAFNet yang dipakai (GoPro-width64 untuk deblur? SIDD untuk denoise?), supaya YAML dan bobot yang tidak dipakai bisa dibuang.
