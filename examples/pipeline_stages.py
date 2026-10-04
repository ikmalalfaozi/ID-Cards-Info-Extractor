"""Contoh penggunaan setiap tahap pipeline, dari gambar mentah sampai JSON.

Jalankan dari root repo:
    python examples/pipeline_stages.py "demo_input/agus suganda.jpg"
    python examples/pipeline_stages.py demo_input/doni.jpg --until preprocess   # tanpa Donut
    python examples/pipeline_stages.py demo_input/doni.jpg --nafnet             # tambah NAFNet

Hasil tiap tahap disimpan di outputs/<nama-gambar>/ (folder ini di-ignore git).
Semua gambar antar tahap berformat RGB (np.ndarray, HWC, uint8).
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # supaya `from src...` jalan tanpa PYTHONPATH

STAGES = ["detect", "corners", "warp", "orientation", "filter", "preprocess", "nafnet", "ocr"]


def save(out_dir, name, rgb):
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"    -> {path}")


# ----------------------------------------------------------------- 1. detect
def stage_detect(rgb):
    """YOLO-seg: temukan dokumen dan mask-nya. Satu gambar bisa berisi >1 dokumen."""
    from src.image_alignment import DocDetector

    detections = DocDetector().detect_and_segment(rgb, conf=0.5)
    if not detections:
        raise SystemExit("Tidak ada dokumen terdeteksi.")
    # contoh pemilihan: ambil dokumen dengan mask terbesar
    best = max(detections, key=lambda d: int(d["mask"].sum()))
    print(f"    {len(detections)} dokumen terdeteksi, bbox terpilih: {best['bbox'].tolist()}")
    return best  # {'bbox': [x1,y1,x2,y2], 'mask': uint8 HxW (0/1)}


# --------------------------------------------------------------- 2. corners
def stage_corners(rgb, mask):
    """DocAligner + mask untuk menentukan 4 sudut (urutan TL, TR, BR, BL).

    `mask` opsional; dipakai sebagai cadangan bila DocAligner memberi <4 titik.
    Gunakan interactive=True untuk menggeser sudut manual (butuh matplotlib GUI).
    """
    from src.image_alignment import CornerDetector

    corners = CornerDetector().detect_corners(rgb, mask=mask, interactive=False)
    print(f"    sudut: {np.round(corners).astype(int).tolist()}")
    return corners


# ------------------------------------------------------------------ 3. warp
def stage_warp(rgb, corners):
    """Transformasi perspektif -> dokumen tampak lurus dari depan."""
    from src.image_alignment import warp_image

    warped = warp_image(rgb, corners)  # order_pts=True bila sudut belum terurut
    print(f"    ukuran hasil warp (H, W): {warped.shape[:2]}")
    return warped


# -------------------------------------------------------------- 4. orientation
def stage_orientation(rgb):
    """Klasifikasi rotasi ('0','90','180','270') lalu putar ke posisi tegak."""
    from src.image_alignment import DocOrientationDetector

    detector = DocOrientationDetector()
    label = detector.detect_orientation(rgb)
    print(f"    orientasi terdeteksi: {label}")
    return detector.correct_orientation(rgb, label)


# ----------------------------------------------------------------- 5. filter
def stage_filter(rgb):
    """Filter jenis dokumen: KTP / SIM / Passport / Other."""
    from ultralytics import YOLO
    from src.utils import classify_document_type

    result = classify_document_type(YOLO(os.path.join(ROOT, "models", "doc-type-cls.pt")), rgb)
    print(f"    jenis: {result['class']} (p={result['probability']:.2f})")
    return result


# ------------------------------------------------------------ 6. preprocess
def stage_preprocess(rgb):
    """Contrast enhancement -> denoising -> deblurring.

    Pilih satu metode per langkah; yang lain tersedia di src/preprocessing/
    (msrcp, equalize_histogram, median/NLM/TV/wavelet, richardson_lucy_deblur, ...).
    """
    from src.preprocessing import apply_clahe, bilateral_filter_denoising, wiener_deblur, create_gaussian_psf

    out = apply_clahe(rgb, clip_limit=2.0, tile_grid_size=(8, 8))
    out = bilateral_filter_denoising(out, d=9, sigma_color=75, sigma_space=75)
    out = wiener_deblur(out, psf=create_gaussian_psf(kernel_size=5, sigma=1.0), balance=0.1)
    if out.dtype != np.uint8:  # beberapa fungsi mengembalikan float 0-1
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------ 7. nafnet
def stage_nafnet(rgb, config="nafnet-options/NAFNet-GoPro-width32.yaml"):
    """(Opsional) restorasi dengan NAFNet sebagai alternatif deblur klasik.

    Bobot diunduh otomatis lewat gdown ke models/ pada pemakaian pertama.
    """
    from src.nafnet.utils import parse, create_model, img2tensor, tensor2img

    opt = parse(os.path.join(ROOT, config))
    opt["dist"] = False  # SEMENTARA: bug di model.py membaca opt['dist'] yang tidak ada
    opt["num_gpu"] = 0 if not _cuda() else opt.get("num_gpu", 1)
    model = create_model(opt)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    model.feed_data({"lq": img2tensor(bgr).unsqueeze(0)})
    model.test()
    out_bgr = tensor2img(model.get_current_visuals()["result"])
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def _cuda():
    import torch
    return torch.cuda.is_available()


# --------------------------------------------------------------------- 8. ocr
def stage_ocr(rgb):
    """Donut: gambar -> JSON field. Model (~777 MB) diunduh ke ./models saat pertama kali."""
    from transformers.utils import logging as hf_logging
    from src.ocr import DonutInfoExtractor

    hf_logging.set_verbosity_error()  # sembunyikan dump config & peringatan processor dari transformers
    return DonutInfoExtractor().predict(Image.fromarray(rgb))


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--until", choices=STAGES, default="ocr", help="berhenti setelah tahap ini selesai")
    ap.add_argument("--nafnet", action="store_true", help="sertakan tahap NAFNet setelah preprocess")
    args = ap.parse_args()

    name = os.path.splitext(os.path.basename(args.image))[0].replace(" ", "_")
    out_dir = os.path.join(ROOT, "outputs", name)
    os.makedirs(out_dir, exist_ok=True)
    stop = STAGES.index(args.until)

    def done(stage):  # True bila pipeline harus berhenti setelah `stage`
        return STAGES.index(stage) >= stop

    rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)

    print("[1] detect")
    det = stage_detect(rgb)
    if done("detect"):
        return

    print("[2] corners")
    corners = stage_corners(rgb, det["mask"])
    if done("corners"):
        return

    print("[3] warp")
    img = stage_warp(rgb, corners)
    save(out_dir, "1_warped.png", img)
    if done("warp"):
        return

    print("[4] orientation")
    img = stage_orientation(img)
    save(out_dir, "2_oriented.png", img)
    if done("orientation"):
        return

    print("[5] filter")
    doc = stage_filter(img)
    if doc["class"] == "Other":
        raise SystemExit("Bukan KTP/SIM/Paspor -> berhenti.")
    if done("filter"):
        return

    print("[6] preprocess")
    img = stage_preprocess(img)
    save(out_dir, "3_preprocessed.png", img)
    if done("preprocess"):
        return

    if args.nafnet:
        print("[7] nafnet")
        img = stage_nafnet(img)
        save(out_dir, "4_nafnet.png", img)
    if done("nafnet"):
        return

    print("[8] ocr")
    print("    ", stage_ocr(img))


if __name__ == "__main__":
    main()
