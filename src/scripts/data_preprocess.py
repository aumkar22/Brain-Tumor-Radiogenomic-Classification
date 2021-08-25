import numpy as np
import cv2
import SimpleITK as sitk


def binarize_mri(input_image: np.ndarray) -> np.ndarray:

    """
    Function to extract brain

    :param input_image: Input MRI scan
    :return: Binarized extracted brain mask
    """

    _, threshold_img = cv2.threshold(input_image, 0, 255, cv2.THRESH_OTSU)
    threshold_img = np.uint8(threshold_img)
    kernel = np.ones((8, 8), np.uint8)
    binary_mask = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)

    return binary_mask


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
