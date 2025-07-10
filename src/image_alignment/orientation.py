import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import gdown


class DocOrientationDetector:
    def __init__(self, google_drive_file_id: str = "1oIJ9KlGXNqvbcYknTWR0zb153Fjhzwua",
                 model_save_path: str = "models/doc-oc.pt"):
        """
        Initialize the document orientation detector.
        Downloads the model from Google Drive if file ID is provided.
        Automatically detects and uses GPU if available, otherwise CPU.

        Args:
            google_drive_file_id (str, optional): Google Drive file ID of the YOLO segmentation model.
            model_save_path (str): Local path where the model will be saved/loaded from.
        """
        self.model_path = model_save_path

        # 1. Determine the device automatically
        if torch.cuda.is_available():
            self.device = "cuda"
            print("ImageOrientationDetector: GPU (CUDA) is available. Using GPU.")
        else:
            self.device = "cpu"
            print("ImageOrientationDetector: GPU (CUDA) is not available. Using CPU.")

        # 2. Download model if file ID is provided and model doesn't exist
        if google_drive_file_id:
            if not os.path.exists(self.model_path):
                print(f"Model not found at {self.model_path}. Attempting to download from Google Drive...")
                try:
                    # Construct Google Drive download URL
                    url = f'https://drive.google.com/uc?id={google_drive_file_id}'
                    gdown.download(url, self.model_path, quiet=False)
                    print(f"Model downloaded successfully to {self.model_path}.")
                except Exception as e:
                    raise RuntimeError(f"Failed to download model from Google Drive (ID: {google_drive_file_id}): {e}")
            else:
                print(f"Model already exists at {self.model_path}. Skipping download")
        else:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"No Google Drive file ID provided, and model not found at {self.model_path}. Please provide a valid file ID or ensure the model file exists.")
            else:
                print(f"Loading existing model from {self.model_path}.")

        # 3. Initalize the YOLO model
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
