from typing import Any, Dict, List, Optional, Sequence
import re

from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from ..config import get_device
from ..model_store import external_revision


_DTYPES = (None, "float32", "float16", "bfloat16")
_QUANTIZE = (None, "int8")


class DonutInfoExtractor:
    def __init__(self, model_name: Optional[str] = None, revision: Optional[str] = None, device: Optional[str] = None,
                 num_beams: int = 3, max_length: Optional[int] = None, dtype: Optional[str] = None,
                 quantize: Optional[str] = None):
        """
        Initialize DonutInfoExtractor with the Donut model.

        By default the model and the pinned commit come from the model manifest (idcard_extractor/model_manifest.json).
        The defaults reproduce the thesis setup (fp32, beam search 3, max_length = model limit); every option below
        trades speed against fidelity and should be validated (e.g. CER) before use.

        Parameters:
        model_name (str, optional): Hugging Face repo id of the Donut model.
        revision (str, optional): Commit/tag to load. Only the default model is pinned automatically;
            for another `model_name` the latest revision is used unless `revision` is given.
        device (str, optional): "cpu", "cuda" or None for automatic (see idcard_extractor.config.get_device).
        num_beams (int): Beam search width (1 = greedy, faster). Can be overridden per call.
        max_length (int, optional): Maximum generated length (default: the model limit). A smaller value stops runaway
            generations early; it cannot exceed the model limit. Can be overridden per call.
        dtype (str, optional): "float32" (default), "bfloat16", or "float16" (CUDA only).
        quantize (str, optional): "int8" = dynamic int8 quantization of Linear layers (CPU only, fp32 only).
            Faster on CPU but changes the outputs.
        """
        if dtype not in _DTYPES:
            raise ValueError(f"dtype tidak dikenal: {dtype!r} (gunakan {', '.join(map(repr, _DTYPES))}).")
        if quantize not in _QUANTIZE:
            raise ValueError(f"quantize tidak dikenal: {quantize!r} (gunakan {', '.join(map(repr, _QUANTIZE))}).")
        self.device = get_device(device)
        if dtype == "float16" and not self.device.startswith("cuda"):
            raise ValueError("dtype='float16' hanya untuk CUDA; di CPU gunakan 'bfloat16' atau biarkan fp32.")
        if quantize and (self.device != "cpu" or dtype not in (None, "float32")):
            raise ValueError("quantize='int8' hanya untuk CPU dengan dtype fp32.")
        self._check_beams(num_beams)

        default_repo, default_revision = external_revision("donut")
        if model_name is None:
            model_name = default_repo
        if revision is None and model_name == default_repo:
            revision = default_revision
        self.model_name = model_name
        self.revision = revision
        self.num_beams = num_beams
        self.max_length = max_length
        self.dtype = dtype
        self.quantize = quantize
        self.processor, self.model = self.download_model(model_name, revision=revision)
        self.model.to(self.device)
        if dtype in ("float16", "bfloat16"):
            import torch
            self._torch_dtype = getattr(torch, dtype)
            self.model.to(self._torch_dtype)
        else:
            self._torch_dtype = None
        if quantize == "int8":
            import torch
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        if max_length is not None:
            self._resolve_max_length(max_length)  # gagal lebih awal bila di luar batas model

    @staticmethod
    def _check_beams(num_beams):
        if not isinstance(num_beams, int) or num_beams < 1:
            raise ValueError(f"num_beams harus bilangan bulat >= 1, diperoleh {num_beams!r}.")

    def _resolve_max_length(self, max_length: Optional[int]) -> int:
        limit = self.model.decoder.config.max_position_embeddings
        value = max_length or self.max_length or limit
        if value > limit:
            raise ValueError(f"max_length={value} melebihi batas model ({limit}).")
        return value

    def preprocess(self, img: Image.Image) -> Dict[str, Any]:
        """
        Pre-process images before passing them to the model.

        Parameters:
        img (Image.Image): Input image.

        Returns:
        Dict[str, Any]: Processed input for the model.
        """
        return self.processor(images=img, return_tensors="pt")

    def predict(self, image: str | Image.Image, task_prompt: str = "<s_id>", num_beams: Optional[int] = None,
                max_length: Optional[int] = None) -> Dict[str, Any]:
        """
        Predict information from KTP images using the Donut model.

        Parameters:
        image (str | Image.Image): Path or PIL Image of the input KTP image.
        task_prompt (str): Task prompt for the Donut model
        num_beams, max_length: Override the instance defaults for this call.

        Returns:
        Dict[str, Any]: Predicted information extracted from the image.
        """
        return self.predict_batch([image], task_prompt, num_beams=num_beams, max_length=max_length)[0]

    def predict_batch(self, images: Sequence[str | Image.Image], task_prompt: str = "<s_id>",
                      num_beams: Optional[int] = None, max_length: Optional[int] = None,
                      batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Predict several images at once (one `generate` call per `batch_size` images; default: all in one call).
        Returns one dict per image, in the same order. Faster than calling `predict` in a loop, mainly on GPU.
        """
        images = [Image.open(i) if isinstance(i, str) else i for i in images]
        if not images:
            return []
        beams = self.num_beams if num_beams is None else num_beams
        self._check_beams(beams)
        length = self._resolve_max_length(max_length)
        step = batch_size or len(images)
        if step < 1:
            raise ValueError(f"batch_size harus >= 1, diperoleh {batch_size!r}.")

        results: List[Dict[str, Any]] = []
        for start in range(0, len(images), step):
            results.extend(self._predict_chunk(images[start:start + step], task_prompt, beams, length))
        return results

    def _predict_chunk(self, images, task_prompt: str, num_beams: int, max_length: int) -> List[Dict[str, Any]]:
        tokenizer = self.processor.tokenizer
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        if self._torch_dtype is not None:
            pixel_values = pixel_values.to(self._torch_dtype)
        prompt_ids = tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        prompt_ids = prompt_ids.repeat(len(images), 1).to(self.device)

        sequences = self.model.generate(
            pixel_values,
            decoder_input_ids=prompt_ids,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=num_beams,
            bad_words_ids=[[tokenizer.unk_token_id]],
        )

        parsed = []
        for sequence in self.processor.batch_decode(sequences):
            sequence = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token
            parsed.append(self.processor.token2json(sequence))
        return parsed

    def download_model(self, model_name: str, revision: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        Download Donut models from Hugging Face.

        Parameters:
        model_name (str): The name of the Donut model to download.
        revision (str, optional): Commit/tag to download (pin).
        cache_dir (str, optional): Cache directory. Default: the Hugging Face cache (HF_HOME).

        Returns:
        Tuple[DonutProcessor, VisionEncoderDecoderModel]: Processor and model objects.
        """
        processor = DonutProcessor.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        model = VisionEncoderDecoderModel.from_pretrained(model_name, revision=revision, cache_dir=cache_dir)
        return processor, model
