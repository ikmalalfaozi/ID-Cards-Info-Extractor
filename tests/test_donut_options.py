"""DonutInfoExtractor: opsi kecepatan (num_beams, max_length, dtype, quantize) dan predict_batch, dengan model palsu."""
from types import SimpleNamespace

import pytest
import torch

import src.ocr.donut as d


class FakeTokenizer:
    pad_token_id, eos_token_id, unk_token_id = 1, 2, 3
    pad_token, eos_token = "<pad>", "</s>"

    def __call__(self, prompt, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.tensor([[7, 8]])}


class FakeProcessor:
    tokenizer = FakeTokenizer()

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def __call__(self, images, return_tensors="pt"):
        images = images if isinstance(images, list) else [images]
        return SimpleNamespace(pixel_values=torch.zeros(len(images), 3, 4, 4))

    def batch_decode(self, sequences):
        return [f"<s_id><k>{int(row[0])}</k></s>" for row in sequences]

    def token2json(self, sequence):
        return {"seq": sequence}


class FakeModel(torch.nn.Module):
    limit = 1536

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.decoder = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=self.limit))
        self.calls, self.moved = [], []

    @classmethod
    def from_pretrained(cls, *a, **kw):
        return cls()

    def to(self, *args, **kw):
        self.moved.append(args)
        return self

    def generate(self, pixel_values, **kw):
        self.calls.append((pixel_values, kw))
        return torch.arange(pixel_values.shape[0]).unsqueeze(1)    # baris ke-i diawali token i


@pytest.fixture
def donut(monkeypatch):
    monkeypatch.setattr(d, "DonutProcessor", FakeProcessor)
    monkeypatch.setattr(d, "VisionEncoderDecoderModel", FakeModel)
    return lambda **kw: d.DonutInfoExtractor(device="cpu", **kw)


IMG = object()


def test_defaults_reproduce_thesis_setup(donut):
    ex = donut()
    ex.predict(IMG)
    _, kw = ex.model.calls[0]
    assert kw["num_beams"] == 3 and kw["max_length"] == 1536 and kw["use_cache"] is True
    assert "output_scores" not in kw and "return_dict_in_generate" not in kw
    assert kw["bad_words_ids"] == [[3]] and kw["decoder_input_ids"].shape == (1, 2)


def test_instance_defaults_and_per_call_overrides(donut):
    ex = donut(num_beams=1, max_length=300)
    ex.predict(IMG)
    assert (ex.model.calls[0][1]["num_beams"], ex.model.calls[0][1]["max_length"]) == (1, 300)
    ex.predict(IMG, num_beams=5, max_length=200)
    assert (ex.model.calls[1][1]["num_beams"], ex.model.calls[1][1]["max_length"]) == (5, 200)


def test_max_length_cannot_exceed_model_limit(donut):
    with pytest.raises(ValueError, match="melebihi batas model"):
        donut(max_length=2000)
    ex = donut()
    with pytest.raises(ValueError, match="melebihi batas model"):
        ex.predict(IMG, max_length=2000)


def test_invalid_beams_rejected(donut):
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="num_beams"):
            donut(num_beams=bad)


def test_predict_batch_keeps_order_and_batches(donut):
    ex = donut()
    out = ex.predict_batch([IMG] * 5, batch_size=2)
    assert len(out) == 5 and len(ex.model.calls) == 3                       # 2 + 2 + 1
    assert [c[0].shape[0] for c in ex.model.calls] == [2, 2, 1]
    assert [o["seq"] for o in out[:2]] == ["<k>0</k>", "<k>1</k>"]         # urutan dalam satu batch
    assert ex.predict_batch([]) == [] and len(ex.model.calls) == 3
    assert len(donut().predict_batch([IMG] * 3)) == 3                       # tanpa batch_size = satu panggilan


def test_predict_is_predict_batch_of_one(donut):
    ex = donut()
    assert ex.predict(IMG) == ex.predict_batch([IMG])[0]


def test_dtype_validation_and_casting(donut):
    with pytest.raises(ValueError, match="dtype tidak dikenal"):
        donut(dtype="int4")
    with pytest.raises(ValueError, match="float16.*CUDA"):
        donut(dtype="float16")                                              # device=cpu
    ex = donut(dtype="bfloat16")
    ex.predict(IMG)
    assert ex.model.calls[0][0].dtype == torch.bfloat16 and (torch.bfloat16,) in ex.model.moved
    assert all(a != (torch.bfloat16,) for a in donut().model.moved)                 # bawaan: tidak diubah dtype-nya


def test_quantize_validation_and_effect(donut):
    with pytest.raises(ValueError, match="quantize tidak dikenal"):
        donut(quantize="int4")
    with pytest.raises(ValueError, match="hanya untuk CPU dengan dtype fp32"):
        donut(quantize="int8", dtype="bfloat16")
    ex = donut(quantize="int8")
    assert "quantized" in type(ex.model.linear).__module__.lower() or "dynamic" in type(ex.model.linear).__module__.lower()
