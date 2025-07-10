import os
from typing import Any, Dict
import re

from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel


class DonutInfoExtractor:
    def __init__(self, model_name: str = "ikmalalfaozi/donut-base-finetuned-ktp-sim-passport-v3"):
        """
        Initialize DonutInfoExtractor with the Donut model.

        Parameters:
        model_name (str): The name of the Donut model to use.
        """
        self.model_name = model_name
        self.processor, self.model = self.download_model(model_name)

    def preprocess(self, img: Image.Image) -> Dict[str, Any]:
        """
        Pre-process images before passing them to the model.

        Parameters:
        img (Image.Image): Input image.

        Returns:
        Dict[str, Any]: Processed input for the model.
        """
        return self.processor(images=img, return_tensors="pt")

    def predict(self, image: str | Image.Image, task_prompt: str="<s_id>") -> Dict[str, Any]:
        """
        Predict information from KTP images using the Donut model.

        Parameters:
        image (str | Image.Image): Path or PIL Image of the input KTP image.
        task_prompt (str): Task prompt for the Donut model

        Returns:
        Dict[str, Any]: Predicted information extracted from the image.
        """
        if isinstance(image, str):
            img = Image.open(image)
        else:
            img = image

        inputs = self.preprocess(img)
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        outputs = self.model.generate(
            inputs.pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=3,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # Remove first task start token

        return self.processor.token2json(sequence)

    def download_model(self, model_name: str, cache_dir: str = "./models"):
        """
        Download Donut models from Hugging Face.

        Parameters:
        model_name (str): The name of the Donut model to download.
        cache_dir (str): Directory to store downloaded models.

        Returns:
        Tuple[DonutProcessor, VisionEncoderDecoderModel]: Processor and model objects.
        """
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        processor = DonutProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        model = VisionEncoderDecoderModel.from_pretrained(model_name, cache_dir=cache_dir)
        return processor, model

