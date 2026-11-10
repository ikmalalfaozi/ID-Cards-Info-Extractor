"""Smoke test lintas lingkungan (Linux, Windows, Colab, Kaggle): `python -m idcard_extractor.smoke`.

Memeriksa impor, device, manifest model, ketersediaan model, lalu menjalankan detektor pada gambar sintetis.
Keluar dengan kode 1 bila ada pemeriksaan yang gagal. Opsi:
  --image PATH   jalankan alur detect -> sudut -> warp -> orientasi pada gambar nyata
  --all-models   pastikan SEMUA model di manifest dapat diperoleh (unduhan besar)
  --donut        muat Donut (~777 MB) dan jalankan satu prediksi singkat
  --offline      hanya memakai cache (tanpa jaringan)
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np

CORE_MODELS = ("doc-seg", "doc-oc", "doc-corner-heatmap-fastvit-sa24")
MODULES = ("config", "model_store", "utils", "image_alignment", "ocr", "preprocessing", "nafnet")


def synthetic_card(height=480, width=640):
    """Latar gelap dengan 'kartu' terang agak miring; cukup untuk menjalankan model, bukan untuk menilai akurasinya."""
    import cv2

    image = np.full((height, width, 3), 40, dtype=np.uint8)
    card = np.array([[120, 100], [520, 120], [500, 380], [100, 360]], dtype=np.int32)
    cv2.fillConvexPoly(image, card, (235, 235, 235))
    cv2.putText(image, "ID CARD", (190, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
    return image


class Runner:
    def __init__(self):
        self.results = []

    def check(self, name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            self.results.append((name, True, detail or "", time.perf_counter() - start))
        except Exception as e:  # smoke test: laporkan apa pun yang gagal, jangan berhenti di yang pertama
            self.results.append((name, False, f"{type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}",
                                 time.perf_counter() - start))

    def report(self):
        for name, ok, detail, secs in self.results:
            print(f"[{'PASS' if ok else 'FAIL'}] {name:24s} {secs:6.1f}s  {detail}")
        failed = [r for r in self.results if not r[1]]
        print(f"\n{len(self.results) - len(failed)}/{len(self.results)} pemeriksaan lulus.")
        return 0 if not failed else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="gambar nyata untuk uji alur penuh")
    ap.add_argument("--all-models", action="store_true", help="periksa semua model di manifest")
    ap.add_argument("--donut", action="store_true", help="muat Donut dan jalankan satu prediksi")
    ap.add_argument("--offline", action="store_true", help="hanya cache, tanpa jaringan")
    args = ap.parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    run = Runner()
    package = __package__ or "idcard_extractor"

    def imports():
        for name in MODULES:
            importlib.import_module(f"{package}.{name}")
        return f"{len(MODULES)} modul"

    def device():
        from . import config
        parts = [f"python {sys.version.split()[0]}", f"device={config.get_device()}"]
        try:
            import torch
            parts.append(f"torch {torch.__version__} cuda={torch.cuda.is_available()}")
        except ImportError:
            parts.append("torch tidak terpasang")
        try:
            import onnxruntime
            parts.append(f"onnxruntime {onnxruntime.__version__} {onnxruntime.get_available_providers()}")
        except ImportError:
            parts.append("onnxruntime tidak terpasang")
        return " | ".join(parts)

    def manifest():
        from . import model_store
        issues = model_store.manifest_issues()
        if issues:
            raise RuntimeError("; ".join(issues))
        return f"{len(model_store.list_models())} model terdaftar"

    def models():
        from . import model_store
        names = list(model_store.list_models()) if args.all_models else list(CORE_MODELS)
        for name in names:
            model_store.ensure_model(name)
        return f"{len(names)} model tersedia"

    image = synthetic_card()

    def corner():
        from .image_alignment import CornerDetector
        corners = CornerDetector().detect_corners(image)
        return f"{len(corners)} sudut pada gambar sintetis"

    def detect():
        from .image_alignment import DocDetector
        return f"{len(DocDetector().detect_and_segment(image, conf=0.5))} dokumen pada gambar sintetis"

    def orientation():
        from .image_alignment import DocOrientationDetector
        return f"orientasi={DocOrientationDetector().detect_orientation(image)}"

    def real_image():
        import cv2
        from .image_alignment import CornerDetector, DocDetector, warp_image
        bgr = cv2.imread(args.image)
        if bgr is None:
            raise FileNotFoundError(args.image)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = DocDetector().detect_and_segment(rgb, conf=0.5)
        if not detections:
            raise RuntimeError("tidak ada dokumen terdeteksi")
        best = max(detections, key=lambda d: int(d["mask"].sum()))
        corners = CornerDetector().detect_corners(rgb, mask=best["mask"])
        if len(corners) != 4:
            raise RuntimeError(f"{len(corners)} sudut terdeteksi, seharusnya 4")
        return f"warp {warp_image(rgb, corners).shape[:2]}"

    def donut():
        from .ocr.donut import DonutInfoExtractor
        result = DonutInfoExtractor(max_length=64).predict(image)
        return f"keluaran bertipe {type(result).__name__}"

    run.check("impor modul", imports)
    run.check("device dan runtime", device)
    run.check("manifest model", manifest)
    run.check("ketersediaan model", models)
    run.check("CornerDetector", corner)
    run.check("DocDetector", detect)
    run.check("DocOrientationDetector", orientation)
    if args.image:
        run.check("alur gambar nyata", real_image)
    if args.donut:
        run.check("Donut", donut)
    return run.report()


if __name__ == "__main__":
    sys.exit(main())
