import numpy as np
import cv2


def get_ksize(sigma):
    # opencv calculates ksize from sigma as
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    # then ksize from sigma is
    # ksize = ((sigma - 0.8)/0.15) + 2.0

    return int(((sigma - 0.8) / 0.15) + 2.0)


def get_gaussian_blur(img, ksize=0, sigma=5):
    # if ksize == 0, then compute ksize from sigma
    if ksize == 0:
        ksize = get_ksize(sigma)

    # Gaussian 2D-kernel can be separable into 2-orthogonal vectors
    # then compute full kernel by taking outer product or simply mul(V, V.T)
    sep_k = cv2.getGaussianKernel(ksize, sigma)

    # if ksize >= 11, then convolution is computed by applying fourier transform
    return cv2.filter2D(img, -1, np.outer(sep_k, sep_k))


def ssr(img, sigma):
    # Single-scale retinex of an image
    # SSR(x, y) = log(I(x, y)) - log(I(x, y)*F(x, y))
    # F = surrounding function, here Gaussian

    return np.log10(img + 1.0) - np.log10(get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)


def color_balance(img, low_per, high_per):
    """Contrast stretch img by histogram equalization with black and white cap"""

    tot_pix = img.shape[1] * img.shape[0]
    # no.of pixels to black-out and white-out
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100

    # channels of image
    ch_list = []
    if len(img.shape) == 2:
        ch_list = [img]
    else:
        ch_list = cv2.split(img)

    cs_img = []
    # for each channel, apply contrast-stretch
    for i in range(len(ch_list)):
        ch = ch_list[i]
        # cumulative histogram sum of channel
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))

        # find indices for blacking and whiting out pixels
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if li == hi:
            cs_img.append(ch)
            continue
        # lut with min-max normalization for [0-255] bins
        lut = np.array([0 if i < li
                        else (255 if i > hi else round((i - li) / (hi - li) * 255))
                        for i in np.arange(0, 256)], dtype='uint8')
        # contrast-stretch channel
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)

    if len(cs_img) == 1:
        return np.squeeze(cs_img)
    elif len(cs_img) > 1:
        return cv2.merge(cs_img)
    return None


def msr(img, sigma_scales=(15, 80, 250), apply_normalization=True):
    # Multi-Scale Retinex of an image
    # MSR(x,y) = sum(weight[i]*SSR(x,y, scale[i])), i = {1..n} scales

    msr_img = np.zeros(img.shape)
    # for each sigma scale compute SSR
    for sigma in sigma_scales:
        msr_img += ssr(img, sigma)

    # divide MSR by weights of each scale
    # here we use equal weights
    msr_img = msr_img / len(sigma_scales)

    # computed MSR could be in range [-k, +l], k and l could be any real value
    # so normalize the MSR image values in range [0, 255]
    if apply_normalization:
        msr_img = cv2.normalize(msr_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    return msr_img


def msrcr(img, sigma_scales=(15, 80, 250), alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
    # Multi-Scale Retinex with Color Restoration
    # MSRCR(x,y) = G * [MSR(x,y)*CRF(x,y) - b], G=gain and b=offset
    # CRF(x,y) = beta*[log(alpha*I(x,y) - log(I'(x,y))]
    # I'(x,y) = sum(Ic(x,y)), c={0...k-1}, k=no.of channels

    img = img.astype(np.float64) + 1.0
    # Multi-Scale retinex and don't normalize the output
    msr_img = msr(img, sigma_scales, apply_normalization=False)
    # Color-restoration function
    crf = beta * (np.log10(alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))
    # MSRCR
    msrcr_img = G * (msr_img * crf - b)
    # normalize MSRCR
    msrcr_img = cv2.normalize(msrcr_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    # color balance the final MSRCR to flat the histogram distribution with tails on both sides
    msrcr_img = color_balance(msrcr_img, low_per, high_per)

    return msrcr_img


def msrcp(img, sigma_scales=(15, 80, 250), low_per=1, high_per=1):
    # Multi-Scale Retinex with Color Preservation
    # Int(x,y) = sum(Ic(x,y))/3, c={0...k-1}, k=no.of channels
    # MSR_Int(x,y) = MSR(Int(x,y)), and apply color balance
    # B(x,y) = MAX_VALUE/max(Ic(x,y))
    # A(x,y) = max(B(x,y), MSR_Int(x,y)/Int(x,y))
    # MSRCP = A*I

    # Intensity image (Int)
    int_img = (np.sum(img, axis=2) / img.shape[2]) + 1.0
    # Multi-Scale Retinex of intensity image (MSR)
    msr_int = msr(int_img, sigma_scales)
    # color balance of MSR
    msr_cb = color_balance(msr_int, low_per, high_per)

    # B = MAX/max(Ic)
    B = 256.0 / (np.max(img, axis=2) + 1.0)
    # BB = stack(B, MSR/Int)
    BB = np.array([B, msr_cb/int_img])
    # A = min(BB)
    A = np.min(BB, axis=0)
    # MSRCP = A*I
    msrcp = np.clip(np.expand_dims(A, 2) * img, 0.0, 255.0)

    return msrcp.astype(np.uint8)


def contrast_stretching(image: np.ndarray, min_out: int = 0, max_out: int = 255) -> np.ndarray:
    """
    Performs contrast stretching on a grayscale or RGB image.

    Args:
        image (np.ndarray): The input image (grayscale or RGB, uint8 type).
        min_out (int): The desired minimum output pixel value (default: 0).
        max_out (int): The desired maximum output pixel value (default: 255).

    Returns:
        np.ndarray: The contrast-stretched image.

    Raises:
        ValueError: If the image is not a NumPy array or has an unsupported number of channels.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input 'image' must be a NumPy array.")

    # Ensure image is of type uint8 for proper pixel value handling
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Determine if it's grayscale or RGB
    if len(image.shape) == 2:  # Grayscale image (H, W)
        is_grayscale = True
        channels = [image]  # Treat the whole image as one channel
    elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image (H, W, 3)
        is_grayscale = False
        channels = cv2.split(image)  # Split into R, G, B channels
    else:
        raise ValueError("Unsupported image format. Expected grayscale (H, W) or RGB (H, W, 3).")

    stretched_channels = []
    for channel in channels:
        min_in = np.min(channel)
        max_in = np.max(channel)

        # Avoid division by zero if all pixels in the channel have the same value
        if max_in == min_in:
            stretched_channel = channel.copy()  # No stretching possible, return as is
            print(
                f"Warning: Min and Max values are the same ({min_in}) for a channel. No stretching applied to this channel.")
        else:
            # Apply the contrast stretching formula
            # Convert to float for calculation to avoid overflow/underflow, then back to uint8
            stretched_channel = ((channel - min_in) * ((max_out - min_out) / (max_in - min_in))) + min_out
            stretched_channel = np.clip(stretched_channel, min_out, max_out).astype(np.uint8)
        stretched_channels.append(stretched_channel)

    if is_grayscale:
        return stretched_channels[0]  # Return the single stretched channel
    else:
        return cv2.merge(stretched_channels)  # Merge stretched R, G, B channels back


def equalize_histogram(image: np.ndarray, color_space_for_rgb: str = 'YUV') -> np.ndarray:
    """
    Performs histogram equalization on a grayscale or RGB image.

    For grayscale images, it applies equalization directly.
    For RGB images, it converts to a different color space (YUV or HSV),
    equalizes the luminance/value channel, and then converts back to RGB.

    Args:
        image (np.ndarray): The input image (grayscale or RGB, uint8 type).
        color_space_for_rgb (str): 'YUV' or 'HSV'. Specifies the color space
                                   to use for RGB image equalization. Default is 'YUV'.

    Returns:
        np.ndarray: The histogram-equalized image.

    Raises:
        ValueError: If the image is not a NumPy array, has an unsupported number of channels,
                    or an invalid color_space_for_rgb is provided.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input 'image' must be a NumPy array.")

    if image.dtype != np.uint8:
        # Convert to uint8 if not already, as equalizeHist expects it
        image = image.astype(np.uint8)

    if len(image.shape) == 2:  # Grayscale image (H, W)
        return cv2.equalizeHist(image)

    elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image (H, W, 3)
        if color_space_for_rgb.upper() == 'YUV':
            # Convert RGB to YUV
            image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            # Equalize the Y (luminance) channel
            image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
            # Convert back to RGB
            equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
            return equalized_image
        elif color_space_for_rgb.upper() == 'HSV':
            # Convert RGB to HSV
            image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            # Equalize the V (value/brightness) channel
            image_hsv[:, :, 2] = cv2.equalizeHist(image_hsv[:, :, 2])
            # Convert back to RGB
            equalized_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
            return equalized_image
        else:
            raise ValueError(f"Invalid color_space_for_rgb: '{color_space_for_rgb}'. Expected 'YUV' or 'HSV'.")
    else:
        raise ValueError("Unsupported image format. Expected grayscale (H, W) or RGB (H, W, 3).")


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to a grayscale or RGB image.

    For grayscale images, it applies CLAHE directly.
    For RGB images, it converts to YUV color space, applies CLAHE to the
    Y (luminance) channel, and then converts back to BGR.

    Args:
        image (np.ndarray): The input image (grayscale or RGB, uint8 type).
        clip_limit (float): Threshold for contrast limiting. Higher values
                            result in more contrast. Default is 2.0.
        tile_grid_size (tuple): Size of the grid for histogram equalization.
                                Input image will be divided into M x N tiles.
                                Default is (8, 8).

    Returns:
        np.ndarray: The CLAHE-enhanced image.

    Raises:
        ValueError: If the image is not a NumPy array, has an unsupported number of channels.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input 'image' must be a NumPy array.")

    if image.dtype != np.uint8:
        # Convert to uint8 if not already, as CLAHE expects it
        image = image.astype(np.uint8)

    # Create a CLAHE object
    # clipLimit: A threshold for contrast limiting. Higher values mean more contrast.
    # tileGridSize: Size of the grid for histogram equalization. Input image is divided into
    #               equal sized rectangular regions.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(image.shape) == 2:  # Grayscale image (H, W)
        return clahe.apply(image)

    elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB image (H, W, 3) (OpenCV reads as BGR)
        # Convert RGB to YUV
        image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        # Apply CLAHE to the Y (luminance) channel
        image_yuv[:, :, 0] = clahe.apply(image_yuv[:, :, 0])

        # Convert back to RGB
        equalized_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
        return equalized_image
    else:
        raise ValueError("Unsupported image format. Expected grayscale (H, W) or RGB (H, W, 3).")
