import numpy as np
import cv2
import SimpleITK as sitk


def brain_extract(input_image: np.ndarray) -> np.ndarray:

    """
    Function to extract brain

    :param input_image: Input MRI scan
    :return: Binarized extracted brain mask
    """

    brain_mask = np.zeros(input_image.shape, np.float)
    brain_mask[input_image > 0] = 1

    return brain_mask


def bias_field_correction(input_image: sitk.Image, mask: sitk.Image) -> sitk.Image:

    """
    Function to perform N4 bias field correction

    :param input_image: Input MRI scan
    :param mask: Mask to specify which pixels are used to estimate the bias-field and suppress
    pixels close to zero
    :return: Corrected image
    """

    bias_field_correction_filter = sitk.N4BiasFieldCorrectionImageFilter()

    return bias_field_correction_filter.Execute(input_image, mask)


def contrast_enhancement(input_image: np.ndarray) -> np.ndarray:

    """
    Contrast enhancement using histogram equalization

    :param input_image: Input image
    :return: Contrast enhanced image
    """

    return cv2.equalizeHist(input_image)


def normalization(input_image: np.ndarray) -> np.ndarray:

    """
    Image normalization by subtracting mean and dividing by standard deviation

    :param input_image: Input image
    :return: Normalized image
    """

    input_image = input_image[input_image > 0.0]
    normalized_image = (input_image - np.mean(input_image)) / np.std(input_image)
    normalized_image[input_image == 0] = 0

    return normalized_image


def sitk_to_numpy(input_image: sitk.Image) -> np.ndarray:

    """
    SimpleITK image to numpy conversion

    :param input_image: SimpleITK image
    :return: Numpy image
    """

    return sitk.GetArrayFromImage(input_image)


def numpy_to_sitk(input_image: np.ndarray) -> sitk.Image:

    """
    Numpy to SimpleITK conversion

    :param input_image: Numpy image array
    :return: SimpleITK image
    """

    return sitk.GetImageFromArray(input_image)
