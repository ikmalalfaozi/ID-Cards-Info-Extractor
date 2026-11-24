"""Pengemasan (pyproject) dan smoke test: paket `idcard_extractor` harus berisi manifest dan runner yang jujur."""
import tomllib
from pathlib import Path

import numpy as np

from idcard_extractor import smoke

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_declares_package_manifest_and_extras():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert cfg["project"]["name"] == "idcard-extractor"
    assert cfg["tool"]["setuptools"]["package-data"]["idcard_extractor"] == ["model_manifest.json", "nafnet/options/*.yaml"]
    assert (ROOT / "idcard_extractor" / "model_manifest.json").is_file()
    extras = cfg["tool"]["setuptools"]["dynamic"]["optional-dependencies"]
    assert set(extras) == {"onnx-cpu", "onnx-gpu", "torch-backend", "demo", "dev"}
    for spec in [cfg["tool"]["setuptools"]["dynamic"]["dependencies"], *extras.values()]:
        for f in spec["file"]:
            assert (ROOT / f).is_file(), f


def test_old_package_name_is_gone():
    assert not (ROOT / "src").exists()


def test_synthetic_card_is_valid_input():
    image = smoke.synthetic_card()
    assert image.shape == (480, 640, 3) and image.dtype == np.uint8 and image.max() > image.min()


def test_runner_reports_failures_and_sets_exit_code(capsys):
    run = smoke.Runner()
    run.check("baik", lambda: "ok")
    run.check("rusak", lambda: 1 / 0)
    assert run.report() == 1
    out = capsys.readouterr().out
    assert "[PASS] baik" in out and "[FAIL] rusak" in out and "ZeroDivisionError" in out and "1/2" in out


def test_runner_exit_code_zero_when_all_pass():
    run = smoke.Runner()
    run.check("a", lambda: None)
    assert run.report() == 0


def test_nafnet_configs_are_packaged_and_resolved_by_name():
    from idcard_extractor.nafnet.utils import config_path, parse
    import pytest

    assert not (ROOT / "nafnet-options").exists()
    opt = parse("NAFNet-GoPro-width32")                       # nama bawaan paket, tanpa path/cwd
    assert opt["path"]["pretrain_model"] == "nafnet-gopro-w32"
    assert config_path("NAFNet-GoPro-width32.yaml") == config_path("NAFNet-GoPro-width32")
    own = ROOT / "idcard_extractor" / "nafnet" / "options" / "NAFSSR-L_2x.yaml"
    assert parse(str(own))["name"] == parse("NAFSSR-L_2x")["name"]   # path eksplisit tetap bekerja
    with pytest.raises(FileNotFoundError, match="Tersedia: .*NAFNet-GoPro-width32"):
        config_path("tidak-ada")


def test_license_files_and_metadata_are_consistent():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert cfg["project"]["license"] == "AGPL-3.0-only"
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("GNU AFFERO GENERAL PUBLIC LICENSE")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for name in ("NAFNet-MIT.txt", "Donut-MIT.txt", "DocAligner-Apache-2.0.txt", "FastViT-Apple.txt"):
        assert (ROOT / "LICENSES" / name).is_file() and name in notices, name


def test_demo_images_are_synthetic_and_reproducible(tmp_path):
    import cv2
    import hashlib
    import subprocess
    import sys

    def build(out):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_images.py"), "--out", str(out)], check=True,
                       capture_output=True)
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.jpg"))}

    first = build(tmp_path / "a")
    assert sorted(first) == ["contoh_ktp.jpg", "contoh_paspor.jpg", "contoh_sim.jpg"]
    assert first == build(tmp_path / "b")                                   # deterministik (seed tetap)
    assert {p.name for p in (ROOT / "demo_input").glob("*.jpg")} == set(first)
    assert all(cv2.imread(str(ROOT / "demo_input" / name)).shape[:2] == (960, 1280) for name in first)
