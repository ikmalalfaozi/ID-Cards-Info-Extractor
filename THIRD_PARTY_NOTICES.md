# Pemberitahuan pihak ketiga

Proyek ini dilisensikan di bawah AGPL-3.0 (lihat `LICENSE`). Dokumen ini mendaftar kode, model, dan pustaka pihak ketiga
beserta lisensinya. Salinan teks lisensi yang wajib dipertahankan ada di folder `LICENSES/`.

Dokumen ini disusun dengan sungguh-sungguh tetapi bukan nasihat hukum. Bagian yang berasal dari pengamatan (bukan dari
pernyataan penulis aslinya) ditandai *belum dikonfirmasi*.

## Kode yang diadaptasi dan ada di repo ini

| Lokasi | Asal | Lisensi | Berkas |
|---|---|---|---|
| `idcard_extractor/nafnet/` (arsitektur NAFNet/NAFSSR, utilitas pemuatan dan konversi citra) | [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet), Copyright (c) 2022 megvii-model. Beberapa fungsi (mis. `default_init_weights`, `flow_warp`, `pixel_unshuffle`, `ordered_yaml`) berasal dari BasicSR, Copyright 2018-2020 BasicSR Authors | MIT (NAFNet) dan Apache-2.0 (BasicSR) | `LICENSES/NAFNet-MIT.txt` memuat keduanya |
| `idcard_extractor/image_alignment/corner_model.py` (praproses dan pascaproses model sudut) | Mereproduksi logika inferensi `heatmap_reg` dan `point_reg` dari [DocsaidLab/DocAligner](https://github.com/DocsaidLab/DocAligner) | Apache-2.0 | `LICENSES/DocAligner-Apache-2.0.txt` |
| `idcard_extractor/image_alignment/polygon_interactor.py` | Tampaknya diadaptasi dari contoh "Polygon Editor" pada galeri Matplotlib *(belum dikonfirmasi)* | Lisensi Matplotlib | https://matplotlib.org/stable/project/license.html |

Perubahan atas kode itu dilakukan di repo ini: NAFNet dan DocAligner dipangkas dan disesuaikan (pemilihan device terpusat, model
dimuat lewat manifest, tanpa kode pelatihan terdistribusi, tanpa paket `docaligner`/`capybara`).

## Model

| Model | Asal | Lisensi | Catatan |
|---|---|---|---|
| Donut (OCR) | Dilatih ulang dari Donut milik Naver Clova IX, Copyright (c) 2022-present NAVER Corp. | MIT | `LICENSES/Donut-MIT.txt`. Bobot hasil fine-tune ada di `ikmalalfaozi/donut-base-finetuned-ktp-sim-passport-v3` |
| DocAligner (4 model ONNX) | Diterbitkan ulang dari DocsaidLab/DocAligner | Kode Apache-2.0; **bobot dan data latih belum dinyatakan penulisnya** | Dipublikasikan dengan teks Apache-2.0 dan atribusi, sebagai asumsi. Backbone FastViT berasal dari [apple/ml-fastvit](https://github.com/apple/ml-fastvit), lisensi Apple di `LICENSES/FastViT-Apple.txt` |
| NAFNet / NAFSSR (7 bobot) | Bobot resmi dari repo NAFNet | MIT untuk repo; **lisensi bobot tidak dinyatakan terpisah** | Diunduh ulang dari mirror Hugging Face milik proyek ini |
| YOLO: segmentasi dokumen, orientasi, jenis dokumen | Dilatih untuk proyek ini dengan `ultralytics` | Diperlakukan mengikuti AGPL-3.0 | Asumsi aman: model yang dilatih dengan Ultralytics umumnya dianggap terikat lisensinya |

Repo model di Hugging Face (`ikmalalfaozi/id-cards-extractor-models`) memuat tabel lisensi per berkas.

## Pustaka yang dipasang (tidak disalin ke repo)

Pustaka berikut dipasang oleh pengguna lewat `pip`, tidak didistribusikan di repo ini. Lisensi menurut metadata paket pada
versi yang diuji:

| Pustaka | Lisensi |
|---|---|
| ultralytics | AGPL-3.0 |
| transformers | Apache-2.0 |
| torch / torchvision | BSD-3-Clause |
| onnxruntime | MIT |
| onnx2torch (opsional) | Apache-2.0 |
| opencv-python | Apache-2.0 (OpenCV); build wheel memuat komponen lain |
| scikit-image, numpy | BSD |
| Pillow | MIT-CMU |
| PyYAML, gdown | MIT |
| sentencepiece | Apache-2.0 |

`ultralytics` berlisensi AGPL-3.0 dan di-import langsung oleh `DocDetector`, `DocOrientationDetector`, dan
`classify_document_type`. Itu alasan proyek ini memakai AGPL-3.0.

## Data

Dataset yang dipakai untuk pengujian tesis (foto KTP/SIM/Paspor) tidak disertakan di repo ini dan tidak boleh dipublikasikan
tanpa memastikan aspek data pribadinya. Gambar di `demo_input/` sintetis (data fiktif), dibuat oleh `scripts/make_demo_images.py`.
