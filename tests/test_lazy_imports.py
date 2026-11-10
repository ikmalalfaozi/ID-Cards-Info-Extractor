"""F4/F5: paket image_alignment tidak boleh bergantung pada DocAligner/matplotlib saat diimpor."""
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_python(code):
    """Jalankan di proses baru (modul yang sudah terimpor di proses pytest tidak boleh memengaruhi hasil)."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=ROOT, capture_output=True, text=True)


def with_block(code):
    return textwrap.dedent(BLOCK) + textwrap.dedent(code)


BLOCK = """
import sys
class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'docaligner', 'capybara', 'matplotlib'}:
            raise ImportError(f'diblokir untuk uji: {name}')
sys.meta_path.insert(0, Block())
"""


def test_image_alignment_imports_without_docaligner_and_matplotlib():
    r = run_python(with_block("""
        from src.image_alignment import DocDetector, DocOrientationDetector, warp_image, order_points
        import src.image_alignment.corner_detector as cd   # modulnya sendiri juga harus bisa diimpor
        print('ok')
    """))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]


def test_corner_detector_gives_actionable_error_without_docaligner():
    r = run_python(with_block("""
        from src.image_alignment import CornerDetector
        try:
            CornerDetector()
        except ImportError as e:
            print('ImportError:', e)
    """))
    assert "ImportError: DocAligner tidak dapat diimpor" in r.stdout and "scripts/install.sh" in r.stdout, r.stdout + r.stderr[-500:]


def test_lazy_names_resolve_when_dependencies_exist():
    pytest.importorskip("matplotlib")
    from src import image_alignment as ia
    assert callable(ia.interactive_get_contour)
    with pytest.raises(AttributeError):
        ia.tidak_ada


def test_order_points_lives_in_image_transform_and_is_reexported():
    from src.image_alignment import order_points
    from src.image_alignment.image_transform import order_points as op2
    assert order_points is op2
    pts = np.array([[10, 90], [90, 10], [10, 10], [90, 90]], dtype="float32")   # acak
    assert order_points(pts).tolist() == [[10, 10], [90, 10], [90, 90], [10, 90]]


def test_interactive_mode_refuses_non_interactive_backend():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from src.image_alignment.corner_detector import interactive_get_contour
    with pytest.raises(RuntimeError, match="backend matplotlib interaktif"):
        interactive_get_contour(np.zeros((4, 2)), np.zeros((20, 20, 3), dtype="uint8"))


def test_nafnet_has_no_absolute_src_imports_or_distributed_code():
    for f in (ROOT / "src" / "nafnet").glob("*.py"):
        text = f.read_text(encoding="utf-8")
        assert "from src." not in text and "import src." not in text, f.name
    model_text = (ROOT / "src" / "nafnet" / "model.py").read_text(encoding="utf-8")
    assert "DistributedDataParallel" not in model_text and "current_device" not in model_text
