import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet


def gaussian_blur_denoising(image, kernel_size=(5, 5), sigmaX=0):
    """
    Apply Gaussian Blur to an image.
    Args:
        image (np.array): Input image (grayscale or color RGB).
        kernel_size (tuple): Filter kernel size (WxH). Must be odd and positive.
        sigmaX (float): Standard deviation of the Gaussian kernel in the X direction.
                        If 0, calculated from kernel_size
    Returns:
         np.array: The image that has been denoised with Gaussian Blur.
    """
    # cv2.GaussianBlur can work on grayscale or color images
    blurred_image = cv2.GaussianBlur(image, kernel_size, sigmaX)
    return blurred_image


def median_blur_denoising(image, kernel_size=5):
    """
    Apply Median Blur to an image.
    Args:
        image (np.array): Input image (grayscale or color RGB)
        kernel_size (int): Filter kernel size. Must be odd and greater than 1.
    Returns:
        np.array: Image that has been denoised with Median Blur.
    """
    # cv2.medianBlur can work on grayscale or color images
    blurred_image = cv2.medianBlur(image, kernel_size)
    return blurred_image


def total_variation_denoising(image: np.ndarray, weight: float = 0.1, n_iter_max: int = 200):
    """
    Applies Total Variation (TV) denoising to an image.
    TV denoising is effective at preserving edges while smoothing noise.

    Args:
        image (np.ndarray): Input image. Can be grayscale (2D array) or color (3D array) RGB.
                            Expected to be in the range [0, 255] for uint8, or [0, 1] for float.
        weight (float): Denoising weight. Higher values lead to more denoising but may
                        smooth out fine details. Typical range is 0.01 to 0.5.
        n_iter_max (int): Maximum number of iterations for the optimization algorithm.

    Returns:
        np.ndarray: Denoised image with the same shape and data type as the input.
                    If input is uint8, the output will also be uint8.
    """
    # Normalize image to [0, 1] for scikit-image's TV denoising
    # Ensure float32 for processing
    img_float = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image

    # Apply TV denoising
    # For color images, denoise_tv_chambolle handles multiple channels automatically
    denoised_img_float = denoise_tv_chambolle(img_float, weight=weight, max_num_iter=n_iter_max, channel_axis=-1 if len(image.shape) == 3 else None)

    # Convert back to original data type and range if input was uint8
    if image.dtype == np.uint8:
        denoised_img = (denoised_img_float * 255).astype(np.uint8)
    else:
        denoised_img = denoised_img_float

    return denoised_img


def bilateral_filter_denoising(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
    """
    Applies Bilateral Filter denoising to an image.
    Bilateral filter is non-linear and effective at reducing noise while
    preserving edges by averaging only pixels with similar intensity and close proximity.

    Args:
        image (np.ndarray): Input image. Can be grayscale (2D array) or color (3D array) RGB.
                            Expected to be in the range [0, 255] for uint8.
        d (int): Diameter of the pixel neighborhood used during filtering.
                 Larger `d` means more pixels are considered, leading to more blurring,
                 but also more computational cost. Use 0 for an automatically determined value.
        sigma_color (float): Filter sigma in the color space. A larger value means that
                             farther colors (in terms of pixel intensity difference)
                             will be mixed together.
        sigma_space (float): Filter sigma in the coordinate space. A larger value means that
                             farther pixels (in terms of spatial distance) will influence
                             each other as long as their colors are close enough.

    Returns:
        np.ndarray: Denoised image with the same shape and data type as the input.
    """
    # Ensure image is in uint8 for OpenCV's bilateralFilter
    if image.dtype != np.uint8:
        # Convert to uint8, scaling if necessary (e.g., from float [0,1] to uint8 [0,255])
        img_uint8 = (image * 255).astype(np.uint8) if np.max(image) <= 1.0 else image.astype(np.uint8)
    else:
        img_uint8 = image

    # Apply bilateral filter
    # OpenCV's bilateralFilter handles grayscale (2D) and BGR (3D) images.
    # If input is RGB, convert to BGR for OpenCV
    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3: # Check if it's a 3-channel image (assumed RGB)
        # OpenCV expects BGR, so convert if the input is RGB
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        denoised_img_bgr = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)
        denoised_img = cv2.cvtColor(denoised_img_bgr, cv2.COLOR_BGR2RGB) # Convert back to RGB
    else: # Grayscale or other channel counts (OpenCV handles these directly)
        denoised_img = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)

    return denoised_img


def wavelet_denoising(image: np.ndarray, sigma: float = None, mode: str = 'soft', wavelet: str = 'db1', rescale_sigma: bool = True):
    """
    Applies Wavelet Denoising to an image.
    Wavelet denoising transforms the image into wavelet coefficients,
    thresholds them to remove noise, and then reconstructs the image.

    Args:
        image (np.ndarray): Input image. Can be grayscale (2D array) or color (3D array) RGB.
                            Expected to be in the range [0, 255] for uint8, or [0, 1] for float.
        sigma (float, optional): The standard deviation of the noise to be removed.
                                 If None, it's estimated from the image (more robust but slower).
        mode (str): Denoising mode. 'soft' or 'hard'. 'soft' thresholding is generally preferred
                    as it produces visually more pleasing results.
        wavelet (str): The type of wavelet to use. E.g., 'db1', 'haar', 'sym2'.
        rescale_sigma (bool): If True, the noise standard deviation sigma is rescaled to match
                              the noise level for each wavelet subband. Recommended for most cases.

    Returns:
        np.ndarray: Denoised image with the same shape and data type as the input.
                    If input is uint8, the output will also be uint8.
    """
    # Normalize image to [0, 1] for scikit-image's wavelet denoising
    img_float = image.astype(np.float32) / 255.0 if image.dtype == np.uint8 else image

    # Apply wavelet denoising
    # For color images, denoise_wavelet handles multiple channels (multichannel=True)
    denoised_img_float = denoise_wavelet(
        img_float,
        sigma=sigma,
        mode=mode,
        wavelet=wavelet,
        rescale_sigma=rescale_sigma,
        channel_axis=-1 if len(image.shape) == 3 else None
    )

    # Convert back to original data type and range if input was uint8
    if image.dtype == np.uint8:
        denoised_img = (denoised_img_float * 255).astype(np.uint8)
    else:
        denoised_img = denoised_img_float

    return denoised_img


def non_local_means_denoising(image: np.ndarray, h: float = 10, templateWindowSize: int = 7, searchWindowSize: int = 21):
    """
    Applies Non-Local Means (NLM) denoising to an image.
    NLM denoising works by averaging all pixels in the image, weighted by how similar
    they are to the pixel being denoised. It's very effective at preserving image details.

    Args:
        image (np.ndarray): Input image. Can be grayscale (2D array) or color (3D array, RGB).
                            Expected to be in the range [0, 255] and of type np.uint8.
        h (float): Parameter regulating the filter strength. A larger `h` value removes more noise
                   but can also remove image detail. (Recommended values: 10 for grayscale,
                   10-20 for color images).
        templateWindowSize (int): Size of the square patch to compare. Should be an odd number.
                                  (Recommended values: 7 for grayscale, 7 or 11 for color).
        searchWindowSize (int): Size of the square patch to search for similar patches.
                                Should be an odd number. (Recommended values: 21).

    Returns:
        np.ndarray: Denoised image with the same shape and data type as the input.
                    Returns None if the input image is not uint8 or has an unsupported number of channels.
    """
    if len(image.shape) == 2:  # Grayscale image (2D array)
        denoised_img = cv2.fastNlMeansDenoising(
            src=image,
            dst=None,
            h=h,
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize
        )
    else: # Color image (3D array)
        # OpenCV's fastNlMeansDenoisingColored expects BGR input.
        # If your input is RGB, convert it first.
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        denoised_img_bgr = cv2.fastNlMeansDenoisingColored(
            src=img_bgr,
            dst=None,
            h=h,
            hColor=h, # hColor is usually the same as h for color denoising
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize
        )
        # Convert back to RGB if the original input was RGB
        denoised_img = cv2.cvtColor(denoised_img_bgr, cv2.COLOR_BGR2RGB)

    return denoised_img
