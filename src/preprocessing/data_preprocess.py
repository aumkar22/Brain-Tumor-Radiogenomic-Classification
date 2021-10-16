import numpy as np

from skimage import exposure


class ImagePreProcess:

    """
    Preprocessing MRI scans
    """

    def __init__(self, input_image: np.ndarray):
        """
        Initialize with input scan

        :param input_image: Input MRI scan
        """

        self.input_image = input_image

    @staticmethod
    def contrast_enhancement(img: np.ndarray) -> np.ndarray:
        """
        Contrast enhancement using histogram equalization

        :param img: Input image
        :return: Contrast enhanced image
        """
        return exposure.equalize_hist(img)

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

        contrast_enhanced_image = self.contrast_enhancement(self.input_image)
        preprocessed_image = self.normalization(contrast_enhanced_image)

        return preprocessed_image
