"""DonutInfoExtractor: opsi kecepatan (num_beams, max_length, dtype, quantize) dan predict_batch, dengan model palsu."""
from types import SimpleNamespace

import pytest
import torch

import idcard_extractor.ocr.donut as d


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


# ------------------------------------------------------------------ stop_on_repeat
def test_repeated_token_stop_judges_each_row():
    stop = d.RepeatedTokenStop(3)
    ids = torch.tensor([[9, 4, 4, 4], [9, 4, 5, 4], [4, 4, 4, 5]])
    assert stop(ids, None).tolist() == [True, False, False]
    assert stop(torch.tensor([[4, 4]]), None).tolist() == [False]            # lebih pendek dari `repeats`


def test_stop_on_repeat_is_off_by_default_and_passes_criteria_when_set(donut):
    ex = donut()
    ex.predict(IMG)
    assert "stopping_criteria" not in ex.model.calls[0][1]                  # keluaran tesis tidak tersentuh
    ex = donut(stop_on_repeat=10)
    ex.predict(IMG)
    crit = ex.model.calls[0][1]["stopping_criteria"]
    assert [c.repeats for c in crit] == [10]
    ex.predict(IMG, stop_on_repeat=12)                                      # per panggilan menimpa bawaan instance
    assert [c.repeats for c in ex.model.calls[1][1]["stopping_criteria"]] == [12]
    ex.predict_batch([IMG, IMG], stop_on_repeat=7)
    assert [c.repeats for c in ex.model.calls[2][1]["stopping_criteria"]] == [7]


def test_stop_on_repeat_validation(donut):
    for bad in (1, 0, -3, 2.5, True, "10"):
        with pytest.raises(ValueError, match="stop_on_repeat"):
            donut(stop_on_repeat=bad)
    with pytest.raises(ValueError, match="stop_on_repeat"):
        donut().predict(IMG, stop_on_repeat=1)


def test_stop_on_repeat_really_shortens_generation_with_transformers():
    """Integrasi dengan generate sungguhan (model MBart kecil acak, greedy): model yang selalu mengulang satu token berhenti dini.

    Pada beam search generate baru berhenti bila SEMUA beam berulang; itu tidak diuji di sini (perlu diukur pada Donut asli).
    """
    beams = 1
    from transformers import LogitsProcessor, LogitsProcessorList, MBartConfig, MBartForCausalLM

    class Force(LogitsProcessor):
        def __call__(self, input_ids, scores):
            scores[:] = -1e9
            scores[:, 5] = 0.0
            return scores

    torch.manual_seed(0)
    model = MBartForCausalLM(MBartConfig(vocab_size=20, d_model=16, decoder_layers=1, decoder_attention_heads=2,
                                         decoder_ffn_dim=16, max_position_embeddings=64, is_decoder=True,
                                         is_encoder_decoder=False)).eval()
    prompt = torch.tensor([[7, 8]])
    common = dict(max_length=60, num_beams=beams, pad_token_id=1, eos_token_id=2, logits_processor=LogitsProcessorList([Force()]))
    full = model.generate(prompt, **common)
    stopped = model.generate(prompt, stopping_criteria=d.StoppingCriteriaList([d.RepeatedTokenStop(10)]), **common)
    assert full.shape[1] == 60
    assert stopped.shape[1] <= 2 + 10 + 1 and (stopped[0, 2:2 + 10] == 5).all()
