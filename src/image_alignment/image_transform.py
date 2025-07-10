import numpy as np
import cv2
from src.image_alignment import order_points


def warp_image(image, pts, output_width=None, output_height=None, order_pts=False):
    """
    Performs a perspective transformation on an image based on 4 corner points.

    Args:
        image (np.ndarray): The source image (NumPy array, BGR or RGB).
        pts (np.ndarray): A 4x2 NumPy array containing the coordinates
                          of the 4 document corner points in the source image.
                          It is recommended that these points are already ordered
                          (TL, TR, BR, BL) using the `order_points` function.
        output_width (int, optional): The desired width of the warped output image.
                                      If None, it will be calculated from the distance between points.
        output_height (int, optional): The desired height of the warped output image.
                                       If None, it will be calculated from the distance between points.
        order_pts (bool): Boolean value to determine whether the points need to be sorted first.

    Returns:
        np.ndarray: The warped (flattened) image.
    """
    # Ensure points are ordered
    rect = pts
    if order_pts:
        rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Calculate the maximum width and height of the new document
    # Width: Distance between tr and tl, or br and bl
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Height: Distance between tr and br, or tl and bl
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # If output_width/height are not specified, use the calculated values
    if output_width is None:
        output_width = maxWidth
    if output_height is None:
        output_height = maxHeight

    # Define the 4 destination points (standard rectangular corners)
    # The order must match the source points: top-left, top-right, bottom-right, bottom-left
    dst = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply the perspective transformation
    warped = cv2.warpPerspective(image, M, (output_width, output_height))

    return warped
