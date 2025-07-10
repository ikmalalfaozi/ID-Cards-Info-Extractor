import numpy as np
import cv2
from ultralytics import YOLO


def classify_document_type(model: YOLO, image: np.ndarray):
    """
    Classifies the type of document in the given image.

    Args:
        model (YOLO): An initialized YOLO classification model.
        image (np.ndarray): Input image (BGR or RGB).

    Returns:
        dict: A dictionary containing:
              - "class": Predicted class name (str)
              - "probability": Confidence score for the predicted class (float)
              - "probabilities": Dictionary of all class probabilities (dict)
    """
    # YOLO's predict method can directly take a NumPy array
    results = model.predict(image, verbose=False) # verbose=False to suppress verbose output

    if not results or not results[0].probs:
        # Handle cases where no predictions are made or probs is empty
        return {
            "class": "Unknown",
            "probability": 0.0,
            "probabilities": {}
        }

    probs = results[0].probs.data.tolist()
    names_dict = results[0].names # Dictionary mapping class IDs to names

    max_prob = max(probs)
    max_index = np.argmax(probs)
    predicted_class = names_dict[max_index]

    all_probabilities = {names_dict[i]: prob for i, prob in enumerate(probs)}

    return {
        "class": predicted_class,
        "probability": max_prob,
        "probabilities": all_probabilities
    }


def draw_polygon_image(
    img: np.ndarray,
    polygon: np.ndarray,
    thickness: int = 3
) -> np.ndarray:

    colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
    export_img = img.copy()
    _polys = polygon.astype(int)
    _polys_roll = np.roll(_polys, 1, axis=0)
    for p1, p2, color in zip(_polys, _polys_roll, colors):
        export_img = cv2.circle(
            export_img, p2, radius=thickness*2,
            color=color, thickness=-1, lineType=cv2.LINE_AA
        )
        export_img = cv2.arrowedLine(
            export_img, p2, p1, color=color,
            thickness=thickness, line_type=cv2.LINE_AA
        )
    return export_img


def add_padding_cv2(image_np, padding_pixels, fill_color=(0, 0, 0), border_type=cv2.BORDER_CONSTANT):
    """
    Adding padding to an image using OpenCV.

    Args:
        image_np (np.ndarray): Image as a NumPy array (usually BGR for OpenCV).
        padding_pixels (int or tuple): The amount of padding in pixels.
                                       - int: uniform padding on all sides.
                                       - (left, top, right, bottom): specific padding for each side.
        fill_color (tuple): Padding color in BGR format (eg: (0, 0, 0) for black, (255, 255, 255) for white).
                            For grayscale images, provide a single int value (0-255).
        border_type (int): Border type. Default cv2.BORDER_CONSTANT (constant color).
                           Another example: cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP.

    Returns:
        np.ndarray: New image with padding as a NumPy array.
    """
    top, bottom, left, right = 0, 0, 0, 0

    if isinstance(padding_pixels, int):
        top = bottom = left = right = padding_pixels
    elif isinstance(padding_pixels, tuple) and len(padding_pixels) == 4:
        left, top, right, bottom = padding_pixels
    else:
        raise ValueError("padding_pixels must be an int or a tuple (left, top, right, bottom).")

    # Make sure fill_color matches the number of image channels (for grayscale vs. BGR/RGB)
    if len(image_np.shape) == 2: # Grayscale image
        if isinstance(fill_color, tuple):
            fill_color_cv2 = fill_color[0] # Take only the first value for grayscale
        else:
            fill_color_cv2 = fill_color
    else: # Color image (BGR or RGB)
        fill_color_cv2 = fill_color

    padded_image_cv2 = cv2.copyMakeBorder(image_np, top, bottom, left, right, border_type, value=fill_color_cv2)
    return padded_image_cv2
