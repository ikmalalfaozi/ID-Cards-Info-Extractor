"""Retinex dengan fast=True harus mirip jalur asli (PSNR tinggi) dan jalur bawaan tidak berubah."""
import numpy as np
import pytest

from idcard_extractor.preprocessing import contrast_enhancement as ce


def make_card(h=270, w=428, seed=0):
    # latar dengan gradien cahaya halus + teks/garis tajam, cukup untuk menguji blur sigma besar
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 90 + 80 * (xx / w) + 30 * (yy / h)
    img = np.stack([base, base * 0.95, base * 0.9], axis=-1)
    for y in range(20, h - 20, 18):
        img[y:y + 4, 30:w - 30] *= 0.35
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


@pytest.mark.parametrize("fn,min_db", [(ce.msr, 40), (ce.msrcr, 38), (ce.msrcp, 38)])
@pytest.mark.parametrize("dark", [False, True])
def test_fast_is_close_to_exact(fn, min_db, dark):
    img = make_card()
    if dark:
        img = (img * 0.3).astype(np.uint8)
    exact, fast = fn(img), fn(img, fast=True)
    assert fast.shape == exact.shape and fast.dtype == exact.dtype == np.uint8
    assert psnr(exact, fast) > min_db


def test_fast_flag_is_opt_in():
    img = make_card(h=120, w=190)
    for fn in (ce.msr, ce.msrcr, ce.msrcp):
        assert np.array_equal(fn(img), fn(img, fast=False))


def test_fast_is_faster():
    import time
    img = make_card(h=540, w=856)
    ce.msr(img, fast=True)
    t0 = time.perf_counter(); ce.msr(img); slow = time.perf_counter() - t0
    t0 = time.perf_counter(); ce.msr(img, fast=True); quick = time.perf_counter() - t0
    assert quick < slow
