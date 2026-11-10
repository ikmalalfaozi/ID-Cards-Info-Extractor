import cv2
import numpy as np

from ..config import get_device, onnx_cuda_available
from .image_transform import order_points  # noqa: F401  (tetap bisa diimpor dari sini untuk kompatibilitas)

_NON_INTERACTIVE_BACKENDS = {"agg", "pdf", "ps", "svg", "cairo", "template"}


def interactive_get_contour(corners, image):
    """
    Displays an interactive image, allowing the user to adjust the corners,
    and scales the obtained corners back to the original image size.
    """

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from . import polygon_interactor as poly_i

    if matplotlib.get_backend().lower() in _NON_INTERACTIVE_BACKENDS:
        raise RuntimeError(
            f"interactive=True membutuhkan backend matplotlib interaktif (saat ini '{matplotlib.get_backend()}', "
            "mis. di Colab/Kaggle atau server tanpa layar). Pakai interactive=False atau jalankan secara lokal.")

    original_height, original_width = image.shape[:2]  # Store original image size

    # Determine maximum display size
    max_display_width = 800  # Adjust to your screen width
    max_display_height = 600  # Adjust to your screen height

    # Calculate scaling ratio
    width_ratio = max_display_width / original_width
    height_ratio = max_display_height / original_height
    scale_ratio = min(width_ratio, height_ratio)

    # Calculate display image size
    display_width = int(original_width * scale_ratio)
    display_height = int(original_height * scale_ratio)

    # Resize image for display
    display_image = cv2.resize(image, (display_width, display_height))

    # Scale initial corners
    display_corners = corners * scale_ratio

    poly = Polygon(display_corners, animated=True, fill=False, color="yellow", linewidth=1.0)
    fig, ax = plt.subplots()
    ax.add_patch(poly)
    ax.set_title(('Drag the corners of the box to the corners of the document. \n'
                  'Close the window when finished.'))
    p = poly_i.PolygonInteractor(ax, poly)
    ax.imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.show()

    new_display_points = p.get_poly_points()[:4]
    new_display_points = np.array([[p] for p in new_display_points], dtype="int32")
    new_display_points = new_display_points.reshape(4, 2)

    # Scale obtained corners back to original size
    new_points = new_display_points / scale_ratio
    new_points = np.array([[p] for p in new_points], dtype="int32")
    return new_points.reshape(4, 2)


class CornerDetector:
    def __init__(self, model_type="heatmap", model_cfg=None, device=None):
        """
        Initialize the corner detector using the DocAligner library.

        DocAligner is imported here (not at module level) so the rest of `image_alignment` works without it.

        Args:
            model_type (str | docaligner.ModelType): "heatmap" or "point".
            model_cfg (str): Configuration for the model (e.g., 'lcnet100', 'fastvit_t8', 'fastvit_sa24').
            device (str, optional): "cpu", "cuda" or None for automatic (see src.config.get_device).
                DocAligner runs on onnxruntime: CUDA is used only when onnxruntime-gpu is installed.
        """
        try:
            from capybara import Backend
            from docaligner import DocAligner, ModelType
        except ImportError as e:
            raise ImportError(
                f"DocAligner tidak dapat diimpor ({e}). Pasang dengan: bash scripts/install.sh "
                "(atau lihat docs/INSTALL.md, bagian DocAligner).") from e

        if isinstance(model_type, str):
            try:
                model_type = ModelType[model_type]
            except KeyError:
                raise ValueError(f"Unsupported model type: {model_type!r} (use 'heatmap' or 'point').") from None
        if model_cfg is None:
            if model_type == ModelType.heatmap:
                model_cfg = 'fastvit_sa24'  # Default for heatmap
            elif model_type == ModelType.point:
                model_cfg = 'lcnet050'  # Default for point
            else:
                raise ValueError("Unsupported model type.")

        self.device = get_device(device)
        use_cuda = self.device.startswith("cuda") and onnx_cuda_available()
        backend = Backend.cuda if use_cuda else Backend.cpu
        if self.device.startswith("cuda") and not use_cuda:
            print("CornerDetector: onnxruntime-gpu tidak terpasang. DocAligner memakai CPU.")
        else:
            print(f"CornerDetector: using {'GPU (CUDA)' if use_cuda else 'CPU'}.")

        # Initialize the model
        self.model = DocAligner(model_type=model_type, model_cfg=model_cfg, backend=backend)

    def detect_corners(self, image, mask=None, interactive=False):
        """
        Detect the four corners of a document in the image.

        Args:
            image (numpy.ndarray): Input image (RGB).
            mask (numpy.ndarray, optional): Binary segmentation mask of the document (0 or 255).
            interactive (bool): Displays an interactive image, allowing the user to adjust the corners.

        Returns:
             Corners (numpy.ndarray): List of corner points [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        """
        corners = self.model(image)
        if mask is not None:
            # Contour detection
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_polygon = None
            detection_method = ""
            if contours:
                # 1. Try cv2.approxPolyDP first
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

                if len(approx_polygon) == 4:
                    detected_polygon = approx_polygon.reshape(-1, 2)
                    detection_method = "approxPolyDP"
                else:
                    # If not 4, try a slightly smaller/larger epsilon.
                    epsilon_alt1 = 0.01 * cv2.arcLength(largest_contour, True)
                    approx_polygon_alt1 = cv2.approxPolyDP(largest_contour, epsilon_alt1, True)

                    epsilon_alt2 = 0.03 * cv2.arcLength(largest_contour, True)
                    approx_polygon_alt2 = cv2.approxPolyDP(largest_contour, epsilon_alt2, True)

                    # Choose the one closest to the 4 corners (with priority 4)
                    potential_polygons = [approx_polygon, approx_polygon_alt1, approx_polygon_alt2]
                    best_candidate = None
                    min_diff = float('inf')

                    for poly in potential_polygons:
                        diff = abs(len(poly) - 4)
                        if diff < min_diff:
                            min_diff = diff
                            best_candidate = poly
                        elif diff == min_diff and len(poly) == 4:  # Prioritize 4 if there are multiple with the same diff
                            best_candidate = poly

                    if best_candidate is not None and len(best_candidate) == 4:
                        detected_polygon = best_candidate.reshape(-1, 2)
                        detection_method = "approxPolyDP"
                    else:
                        # 2. Fallback ke cv2.minAreaRect
                        rect = cv2.minAreaRect(largest_contour)
                        corners_float = cv2.boxPoints(rect)
                        detected_polygon = corners_float
                        detection_method = "minAreaRect"

                if len(corners) != 4 and len(detected_polygon) == 4:
                    corners = detected_polygon
                elif len(corners) == 4 and len(detected_polygon) == 4 and detection_method == "approxPolyDP":
                    area_docaligner = cv2.contourArea(corners.astype(np.float32))
                    area_contour = cv2.contourArea(detected_polygon.astype(np.float32))
                    if area_contour > area_docaligner:
                        corners = detected_polygon

        if len(corners) == 4:
            corners = order_points(corners)
        if interactive:
            corners = interactive_get_contour(np.array(corners), image)
            corners = corners.astype(np.float32)

        return corners
