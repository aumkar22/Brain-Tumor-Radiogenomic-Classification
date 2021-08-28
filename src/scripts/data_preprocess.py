import cv2

from src.util.type_conversions import *


class ImagePreProcess(object):

    """
    Preprocessing MRI scans
    """

    def __init__(self, input_image: sitk.Image):
        """
        Initialize with input scan

        :param input_image: Input MRI scan
        """

        self.input_image = input_image

    @staticmethod
    def brain_extract(img: np.ndarray) -> np.ndarray:
        """
        Function to extract brain

        :param img: Input image
        :return: Binarized extracted brain mask
        """

        brain_mask = np.zeros(img.shape, np.float)
        brain_mask[img > 0] = 1

        return brain_mask

    def bias_field_correction(self, mask: sitk.Image) -> sitk.Image:
        """
        Function to perform N4 bias field correction

        :param mask: Mask to specify which pixels are used to estimate the bias-field and suppress
        pixels close to zero
        :return: Corrected image
        """

        bias_field_correction_filter = sitk.N4BiasFieldCorrectionImageFilter()

        return bias_field_correction_filter.Execute(self.input_image, mask)

    @staticmethod
    def contrast_enhancement(img: np.ndarray) -> np.ndarray:
        """
        Contrast enhancement using histogram equalization

        :param img: Input image
        :return: Contrast enhanced image
        """

        return cv2.equalizeHist(img)

    @staticmethod
    def normalization(img: np.ndarray) -> np.ndarray:
        """
        Image normalization by subtracting mean and dividing by standard deviation

        :param img: Input image
        :return: Normalized image
        """

        img = img[img > 0.0]
        normalized_image = (img - np.mean(img)) / np.std(img)
        normalized_image[img == 0] = 0

        return normalized_image

    def apply_preprocess(self) -> np.ndarray:
        """
        Apply preprocessing to the scan

        :return: Preprocessed scan
        """

        numpy_image = sitk_to_numpy(self.input_image)
        brain_mask = self.brain_extract(numpy_image)
        sitk_mask = numpy_to_sitk(brain_mask)
        bias_field_corrected_image = self.bias_field_correction(sitk_mask)
        contrast_enhanced_image = self.contrast_enhancement(
            sitk_to_numpy(bias_field_corrected_image)
        )
        preprocessed_image = self.normalization(contrast_enhanced_image)

        return preprocessed_image
