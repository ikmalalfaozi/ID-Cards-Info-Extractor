import cv2
import numpy as np
from skimage.restoration import wiener, richardson_lucy
import os


def wiener_deblur(image_input, psf: np.ndarray, balance: float, is_real: bool = True, clip: bool = True):
    """
    Deblurring images using scikit-image's Wiener Filter.
    Supports image input from file paths, grayscale NumPy arrays, or RGB NumPy arrays.

    Args:
        image_input (str or np.ndarray): Path to the image file or a grayscale or RGB
        psf (np.ndarray): Point Spread Function (PSF) of blur.
        balance (float): Regularization parameters to balance frequency restoration and noise artifact reduction.
        is_real (bool, optional): True if PSF and reg are provided with hermitian hypothesis. Default is True.
        clip (bool, optional): True if the result pixel value is above 1 or below -1 will be clipped. Default is True.

    Returns:
        np.ndarray: Deblurred image (grayscale or RGB).
    """
    original_is_rgb = False
    img_processed = None

    # --- 1. Handle Different Input Types image_input ---
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            print(f"Error: File '{image_input}' not found.")
            return None
        img = cv2.imread(image_input)
        if img is None:
            print(f"Error: Unable to read image from '{image_input}'.")
            return None

        original_is_rgb = True
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycbcr)
        img_to_deblur = Y.astype(np.float32) / 255.0
        chroma_channels = (Cr, Cb)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            original_is_rgb = True
            img_ycbcr = cv2.cvtColor(image_input, cv2.COLOR_RGB2YCrCb)
            Y, Cr, Cb = cv2.split(img_ycbcr)
            img_to_deblur = Y.astype(np.float32) / 255.0
            chroma_channels = (Cr, Cb)
        elif len(image_input.shape) == 2:
            img_to_deblur = image_input.astype(np.float32) / 255.0
        else:
            print("Error: Unsupported image input format.")
            return None
    else:
        print("Error: Unsupported image input format.")
        return None

    # --- 2. Do Deblurring using Wiener Filter ---
    img_deconv_y = wiener(img_to_deblur, psf, balance=balance, is_real=is_real, clip=clip)
    img_deconv_y = np.clip(img_deconv_y * 255, 0, 255).astype(np.uint8)

    # --- 3. Color Image Reconstruction (if input is RGB) ---
    if original_is_rgb:
        img_processed = cv2.merge([img_deconv_y, *chroma_channels])
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_YCrCb2RGB)
    else:
        img_processed = img_deconv_y

    return img_processed


def richardson_lucy_deblur(image_input, psf: np.ndarray, num_iter: int, clip: bool = True):
    """
    Deblurring images using scikit-image's Richardson-Lucy.
    Supports image input from file paths, grayscale NumPy arrays, or RGB NumPy arrays.

    Args:
        image_input (str or np.ndarray): Path to the image file or a grayscale or RGB
        psf (np.ndarray): Point Spread Function (PSF) of blur.
        num_iter (int): Number of iterations for Richardson-Lucy.
        clip (bool, optional): True if the result pixel value is above 1 or below -1 will be clipped. Default is True.

    Returns:
        np.ndarray: Deblurred image (grayscale or RGB).
    """
    original_is_rgb = False
    img_processed = None

    # --- 1. Handle Different Input Types image_input ---
    if isinstance(image_input, str):
        if not os.path(image_input):
            print(f"Error: File '{image_input}' not found.")
            return None
        img = cv2.imread(image_input)
        if img is None:
            print(f"Error: Unable to read image from '{image_input}'.")
            return None

        original_is_rgb = True
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycbcr)
        img_to_deblur = Y.astype(np.float32) / 255.0
        chroma_channels = (Cr, Cb)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3:
            original_is_rgb = True
            img_ycbcr = cv2.cvtColor(image_input, cv2.COLOR_RGB2YCrCb)
            Y, Cr, Cb = cv2.split(img_ycbcr)
            img_to_deblur = Y.astype(np.float32) / 255.0
            chroma_channels = (Cr, Cb)
        elif len(image_input.shape) == 2:
            img_to_deblur = image_input.astype(np.float32) / 255.0
        else:
            print("Error: Unsupported image input format.")
            return None
    else:
        print("Error: Unsupported image input format.")
        return None

    # --- 2. Do Deblurring using Richardson-Lucy ---
    img_deconv_y = richardson_lucy(img_to_deblur, psf, num_iter=num_iter, clip=clip)
    img_deconv_y = np.clip(img_deconv_y * 255, 0, 255).astype(np.uint8)

    # --- 3. Color Image Reconstruction (if input is RGB) ---
    if original_is_rgb:
        img_processed = cv2.merge([img_deconv_y, *chroma_channels])
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_YCrCb2RGB)
    else:
        img_processed = img_deconv_y

    return img_processed


def create_gaussian_psf(kernel_size: int, sigma: float):
    x = np.arange(0, kernel_size) - (kernel_size - 1) / 2
    gaussian_1d = np.exp(-(x ** 2) / (2 * sigma ** 2))
    gaussian_1d /= np.sum(gaussian_1d)
    psf = np.outer(gaussian_1d, gaussian_1d)
    psf /= np.sum(psf)
    return psf


def create_average_psf(kernel_size: int):
    psf = np.ones((kernel_size, kernel_size), dtype=np.float32)
    psf /= np.sum(psf)
    return psf


def create_motion_psf(length: int, angle: int):
    psf = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2

    if angle == 0:
        psf[:, int(center)] = 1
    elif angle == 90:
        psf[int(center), :] = 1
    else:
        theta = np.deg2rad(angle)
        x_start, y_start = center - length / 2 * np.cos(theta), center - length / 2 * np.sin(theta)
        x_end, y_end = center + length / 2 * np.cos(theta), center + length / 2 * np.sin(theta)
        cv2.line(psf, (int(x_start), int(y_start)), (int(x_end), int(y_end)), 1, 1)

    psf /= np.sum(psf)
    return psf
