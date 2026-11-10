"""Pemilihan device terpusat (src.config) dan penerusannya ke kelas-kelas."""
import warnings

import pytest
import torch

from src import config


def test_auto_follows_cuda_availability(monkeypatch):
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert config.get_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert config.get_device() == "cpu"


def test_argument_beats_environment(monkeypatch):
    monkeypatch.setenv(config.ENV_DEVICE, "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert config.get_device("cpu") == "cpu"
    assert config.get_device() == "cuda"


def test_environment_can_force_cpu_even_with_gpu(monkeypatch):
    monkeypatch.setenv(config.ENV_DEVICE, "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert config.get_device() == "cpu"


def test_cuda_requested_without_gpu_falls_back_with_warning(monkeypatch):
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="CUDA tidak tersedia"):
        assert config.get_device("cuda") == "cpu"
    with pytest.warns(UserWarning):
        assert config.get_device("cuda:1") == "cpu"


def test_indexed_cuda_is_kept_when_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert config.get_device("cuda:1") == "cuda:1"


def test_unknown_device_is_an_error():
    with pytest.raises(ValueError, match="tidak dikenal"):
        config.get_device("tpu")


def test_auto_and_empty_do_not_warn(monkeypatch):
    monkeypatch.delenv(config.ENV_DEVICE, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert config.get_device("auto") == "cpu"
        assert config.get_device("") == "cpu"
