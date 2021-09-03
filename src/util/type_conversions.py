import SimpleITK as sitk
import numpy as np


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
