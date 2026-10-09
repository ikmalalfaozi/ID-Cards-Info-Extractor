import cv2
import numpy as np
import torch.cuda
from ultralytics import YOLO

from ..model_store import resolve_model_path


class DocDetector:
    def __init__(self, google_drive_file_id=None, model_save_path=None, *, model_path=None):
        """
        Initialize the document detector.
        By default the model ("doc-seg") is resolved through the model manifest (Hugging Face, pinned
        revision, checksum verified). Automatically detects and uses GPU if available, otherwise CPU.

        Args:
            google_drive_file_id: Deprecated and ignored (kept for backward compatibility).
            model_save_path: Alias of `model_path` (backward compatibility).
            model_path (str, optional): Local YOLO weights to use instead of the managed model.
        """
        self.model_path = str(resolve_model_path("doc-seg", model_path=model_path, model_save_path=model_save_path,
                                                 google_drive_file_id=google_drive_file_id))

        # 1. Determine the device automatically
        if torch.cuda.is_available():
            self.device = "cuda"
            print("DocDetector: GPU (CUDA) is available. Using GPU.")
        else:
            self.device = "cpu"
            print("DocDetector: GPU (CUDA) is not available. Using CPU.")

        # 2. Initialize the YOLO model
        try:
            self.model = YOLO(self.model_path)
            print(f"YOLO model initialized successfully with model loaded from {self.model_path}.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize YOLO model from {self.model_path}: {e}")

    def detect_and_segment(self, image_input, conf=0.5):
        """
        Perform detection and segmentation of the document area.

        Args:
            image_input (str or np.ndarray): Path to the input image (string) or the input image as a Numpy array (HWC, RGB).
            conf (float): Confidence score of prediction [0,1].

        Returns:
            list of dict: List of dictionaries, each containing detection and segmentation results (coordinates of the bounding box and binary segmentation mask).
        """
        if isinstance(image_input, str):
            # Input is a file path
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Image not found at path: {image_input}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            source_for_predict = image_rgb
        elif isinstance(image_input, np.ndarray):
            # Input is a Numpy array (assume it's already RGB as per requirement)
            if image_input.ndim != 3 or image_input.shape[2] != 3:
                raise ValueError("Input NumPy array must be a 3-channel (RGB) image.")
            image_rgb = image_input
            source_for_predict = image_rgb
        else:
            raise TypeError("image_input must be a string (file path) or a NumPy array (HWC, RGB).")

        # Perform prediction using the determined device
        results = self.model.predict(source=source_for_predict, conf=conf, device=self.device, verbose=False)

        detections = []
        detection_data = results[0]
        boxes = detection_data.boxes.xyxy.cpu().numpy()
        original_image_height, original_image_width = image_rgb.shape[:2]
        for i, box in enumerate(boxes):
            bbox = box.astype(int)
            current_mask = None

            if detection_data.masks is not None:
                current_mask = np.zeros((original_image_height, original_image_width, 3), dtype=np.uint8)
                # Plot mask
                current_mask = detection_data[i].plot(boxes=False, img=current_mask, labels=False, masks=True)
                current_mask = cv2.cvtColor(current_mask, cv2.COLOR_RGB2GRAY)
                # Apply a threshold to make it a binary mask (0 or 1)
                current_mask = (current_mask >= 1).astype(np.uint8)

            detections.append({
                'bbox': bbox,
                'mask': current_mask
            })

        return detections

