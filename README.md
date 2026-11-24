# ID Cards Info Extractor

Ekstraksi informasi dari foto KTP, SIM, dan Paspor Indonesia menjadi JSON. Foto mentah (miring, buram, ada latar) melewati
deteksi dokumen, perataan perspektif, koreksi rotasi, perbaikan kualitas citra, lalu dibaca oleh model OCR
tanpa-detektor-teks (Donut).

Dikembangkan sebagai bagian dari Tugas Akhir di ITB, "Pengembangan Sistem untuk Ekstraksi Informasi dari ID Card".

## Alur

```
foto mentah
  -> DocDetector (YOLO segmentasi)         temukan dokumen dan mask-nya
  -> CornerDetector (DocAligner, ONNX)     empat sudut dokumen
  -> warp_image                            dokumen tampak lurus dari depan
  -> DocOrientationDetector (YOLO)         putar ke posisi tegak (0/90/180/270)
  -> classify_document_type (YOLO)         KTP / SIM / Passport / Other
  -> preprocessing                         CLAHE, denoise, deblur (atau NAFNet)
  -> DonutInfoExtractor                    JSON: nama, nomor identitas, dst.
```

Setiap tahap berdiri sendiri dan bisa dipakai tanpa tahap lain. Contoh per tahap ada di
[`examples/pipeline_stages.py`](examples/pipeline_stages.py).

## Pasang

Python 3.10-3.13. Dari root repo:

```bash
pip install -e ".[onnx-cpu]"        # atau ".[onnx-gpu]"
python -m idcard_extractor.smoke    # memeriksa impor, device, model, dan detektor
```

Model (alur dasar sekitar 1 GB, sebagian besar Donut) diunduh otomatis dari Hugging Face pada pemakaian pertama dan diverifikasi sha256-nya.
torch/torchvision tidak dipasang oleh perintah di atas; di Colab dan Kaggle sudah tersedia, di tempat lain pasang
dulu (lihat [`docs/INSTALL.md`](docs/INSTALL.md), termasuk Windows, Colab, Kaggle, dan variabel lingkungan).

## Pakai

Gambar di `demo_input/` sintetis (data fiktif, dibuat oleh `scripts/make_demo_images.py`), jadi tidak ada data pribadi.
Pengklasifikasi jenis dokumen menilainya "Other" karena bukan dokumen asli, sehingga contoh baris perintah memakai
`--skip-filter`; untuk foto KTP/SIM/Paspor sungguhan flag itu tidak perlu. Hasil OCR pada gambar sintetis hanya indikatif.

```bash
python examples/pipeline_stages.py demo_input/contoh_ktp.jpg --skip-filter
python examples/pipeline_stages.py demo_input/contoh_ktp.jpg --skip-filter --until preprocess   # tanpa Donut
```

Dari Python:

```python
import cv2
from PIL import Image
from idcard_extractor.image_alignment import DocDetector, CornerDetector, DocOrientationDetector, warp_image
from idcard_extractor.ocr import DonutInfoExtractor

rgb = cv2.cvtColor(cv2.imread("demo_input/contoh_ktp.jpg"), cv2.COLOR_BGR2RGB)

doc = max(DocDetector().detect_and_segment(rgb, conf=0.5), key=lambda d: int(d["mask"].sum()))
corners = CornerDetector().detect_corners(rgb, mask=doc["mask"])
card = warp_image(rgb, corners)

orientation = DocOrientationDetector()
card = orientation.correct_orientation(card, orientation.detect_orientation(card))

print(DonutInfoExtractor().predict(Image.fromarray(card)))   # dict berisi field dokumen
```

Gambar antar tahap berupa `np.ndarray` RGB (HWC, uint8). Field hasilnya bergantung jenis dokumen: KTP (mis. `nomor_identitas`, `nama_lengkap`,
`tempat_tanggal_lahir`), SIM (`nomor_sim`, `golongan_sim`), Paspor (`nomor_paspor`, `jenis_paspor`). Contoh di atas melewati tahap preprocessing; tahap itu ada di contoh per tahap.

Device dipilih otomatis (CUDA bila ada); paksa CPU dengan `IDCARD_DEVICE=cpu`. Donut punya opsi kecepatan
(`num_beams`, `max_length`, `dtype`, `quantize`, `predict_batch`, `stop_on_repeat`), semuanya mati secara bawaan agar sama dengan
setelan tesis. Pengaruhnya pada akurasi perlu divalidasi sebelum dipakai; lihat bagian "Kecepatan Donut" di `docs/INSTALL.md`.

## Isi repo

| Lokasi | Isi |
|---|---|
| `idcard_extractor/image_alignment/` | detektor dokumen, sudut (DocAligner lewat onnxruntime), orientasi, warp |
| `idcard_extractor/preprocessing/` | contrast enhancement, denoising, deblurring klasik |
| `idcard_extractor/nafnet/` | NAFNet untuk restorasi (konfigurasi di `nafnet/options/`) |
| `idcard_extractor/ocr/` | `DonutInfoExtractor` |
| `idcard_extractor/model_store.py`, `model_manifest.json` | unduhan model, dipin ke commit dan sha256 |
| `examples/`, `notebooks/` | contoh per tahap; notebook smoke test Colab/Kaggle |
| `scripts/` | `install.sh`, `upload_models.py` (merilis model baru) |
| `docs/` | instalasi, rancangan penyimpanan model, rancangan portabilitas |
| `tests/` | tes `pytest` (tanpa jaringan dan tanpa file model sungguhan) |

```bash
pip install -e ".[dev]" && pytest
```

## Status pengujian

Diuji: Linux/WSL (Python 3.11 dan 3.13, CPU) dan Kaggle (Tesla T4, Python 3.13). Belum diuji: Windows, Colab,
`onnxruntime-gpu`, dan kesetaraan penuh dengan hasil tesis pada seluruh dataset (2.573 gambar). Rinciannya ada di
`docs/INSTALL.md`.

## Lisensi

Kode proyek ini dilisensikan di bawah **GNU AGPL-3.0** (lihat [`LICENSE`](LICENSE)), Copyright (C) 2025-2026 Ikmal Alfaozi. Alasannya:
`ultralytics` berlisensi AGPL-3.0 dan di-import langsung oleh detektor dokumen. Menjalankan versi yang sudah diubah sebagai layanan
jaringan mewajibkan penyediaan kode sumbernya kepada penggunanya.

Kode, model, dan pustaka pihak ketiga serta lisensinya didaftar di [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); teks lisensi yang
wajib dipertahankan ada di [`LICENSES/`](LICENSES). Ringkasnya:

- **Model** ada di Hugging Face: [`ikmalalfaozi/id-cards-extractor-models`](https://huggingface.co/ikmalalfaozi/id-cards-extractor-models)
  (YOLO, NAFNet, DocAligner) dan `ikmalalfaozi/donut-base-finetuned-ktp-sim-passport-v3` (Donut).
- **DocAligner:** kode asli berlisensi Apache-2.0 ([DocsaidLab/DocAligner](https://github.com/DocsaidLab/DocAligner)). Bobotnya
  diterbitkan ulang dengan atribusi, tetapi penulis aslinya belum menyatakan lisensi untuk bobot maupun data latihnya.
- **NAFNet dan Donut** berlisensi MIT; bobot NAFNet tidak dinyatakan lisensinya secara terpisah.
- Dataset tesis tidak disertakan.
