import cv2
import numpy as np
import torch
from ultralytics import YOLO

from ..model_store import resolve_model_path


class DocOrientationDetector:
    def __init__(self, google_drive_file_id=None, model_save_path=None, *, model_path=None):
        """
        Initialize the document orientation detector.
        By default the model ("doc-oc") is resolved through the model manifest (Hugging Face, pinned
        revision, checksum verified). Automatically detects and uses GPU if available, otherwise CPU.

        Args:
            google_drive_file_id: Deprecated and ignored (kept for backward compatibility).
            model_save_path: Alias of `model_path` (backward compatibility).
            model_path (str, optional): Local YOLO weights to use instead of the managed model.
        """
        self.model_path = str(resolve_model_path("doc-oc", model_path=model_path, model_save_path=model_save_path,
                                                 google_drive_file_id=google_drive_file_id))

        # 1. Determine the device automatically
        if torch.cuda.is_available():
            self.device = "cuda"
            print("ImageOrientationDetector: GPU (CUDA) is available. Using GPU.")
        else:
            self.device = "cpu"
            print("ImageOrientationDetector: GPU (CUDA) is not available. Using CPU.")

        # 2. Initialize the YOLO model
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLO model initialized successfully with model loaded from {self.model_path}.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLO model from {self.model_path}: {e}")

    def detect_orientation(self, image_input):
        """
        Detect the orientation of the image.

        Args:
            image_input (str or np.ndarray): Input image as a file path (str) or a NumPy array (HWC, RGB or BGR).

        Returns:
            str: Predicted orientation ('0', '90', '180', '270').

        Raises:
            ValueError: If the image input is invalid or no orientation is predicted.
            TypeError: If image_input type is not supported.
        """
        if isinstance(image_input, str):
            # Input is a file path
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Image not found at path: {image_input}")
            # YOLO's predict method can generally handle BGR images directly
        elif isinstance(image_input, np.ndarray):
            # Input is a NumPy array
            if image_input.ndim != 3:
                raise ValueError("Input NumPy array must be a 3-channel (HWC) image.")
            image = image_input
        else:
            raise TypeError("image_input must be a string (file path) or a NumPy array.")

        results = self.model.predict(image, device=self.device, verbose=False)

        if not results or not results[0].probs:
            raise ValueError("No orientation prediction found for the image.")

        predicted_label = results[0].names[results[0].probs.top1]
        return predicted_label

    def correct_orientation(self, image_input, orientation: str):
        """
        Rotate the image to the correct orientation based on the predicted label.

        Args:
            image_input (str or np.ndarray): Input image as a file path (str) or a NumPy array (HWC, RGB or BGR).
            orientation (str): Predicted orientation ('0', '90', '180', '270'),
                               representing degrees counter-clockwise from upright.
                               (e.g., '90' means original image is rotated 90 degrees CCW from upright).

        Returns:
            numpy.ndarray: Correctly oriented image.

        Raises:
            ValueError: If the image input is invalid or orientation label is unknown.
            TypeError: If image_input type is not supported.
        """
        # Ensure image is loaded as np.ndarray if input is a path
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Image not found at path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim != 3:
                raise ValueError("Input NumPy array must be a 3-channel (HWC) image.")
            image = image_input
        else:
            raise TypeError("image_input must be a string (file path) or a NumPy array.")

        if orientation == '0':
            return image  # No rotation needed, image is already upright
        elif orientation == '90':
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == '180':
            return cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == '270':
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError(f"Invalid orientation label: {orientation}. Expected '0', '90', '180', or '270'.")
