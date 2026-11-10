"""Image alignment: deteksi dokumen, sudut, warp, dan orientasi.

`CornerDetector` dan `interactive_get_contour` memuat DocAligner/matplotlib, jadi diimpor malas (saat pertama dipakai):
`import src.image_alignment` tetap berhasil walaupun DocAligner belum terpasang.
"""
from .doc_detector import DocDetector
from .image_transform import warp_image, order_points
from .orientation import DocOrientationDetector

__all__ = ["DocDetector", "DocOrientationDetector", "CornerDetector", "interactive_get_contour", "warp_image", "order_points"]


def __getattr__(name):
    if name in ("CornerDetector", "interactive_get_contour"):
        from . import corner_detector
        return getattr(corner_detector, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
