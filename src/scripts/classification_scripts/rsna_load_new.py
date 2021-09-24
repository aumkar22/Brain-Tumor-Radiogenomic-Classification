import pydicom as dicom
import numpy as np
import cv2
import imutils

from pathlib import Path

from src.preprocessing.data_preprocess import *


# from src.scripts.data_load import *


class RsnaLoad:
    def __init__(
        self,
        data_path: Path,
        patient: str,
        label: int,
        num_images: int = 30,
        resize_shape: int = 240,
    ):
        self.data_path = data_path
        self.patient = patient
        self.label = label
        self.num_images = num_images
        self.resize_shape = resize_shape

    @staticmethod
    def dicom_load(dicom_file: str) -> np.ndarray:
        """

        :param dicom_file:
        :return:
        """

        return dicom.read_file(dicom_file).pixel_array

    def data_load(self):
        """

        :return:
        """
        dicom_path = Path(self.data_path / f"{self.patient}" / "FLAIR")
        dicom_list = [str(i) for i in dicom_path.glob(r"**/*.dcm")]
        dicom_list.sort(key=lambda x: int(x.split("-")[-1][:-4]))
        middle_dicom = len(dicom_list) // 2
        divide_images = self.num_images // 2

        left_end = max(0, middle_dicom - divide_images)
        right_end = min(len(dicom_list), middle_dicom + divide_images)
        dicom_data = []
        for dicom_index, dicom_file in enumerate(dicom_list[left_end:right_end]):
            dicom_data.append(self.dicom_load(dicom_file))

        dicom_volume = np.stack(dicom_data)
        preprocessed_dicom_volume = self.perform_preprocess(dicom_volume)
        cropped_dicom = self.crop_slice(
            preprocessed_dicom_volume, preprocessed_dicom_volume > 0
        )

        resize_cropped_dicom = np.expand_dims(
            np.array(
                [
                    cv2.resize(dicom_slice, (self.resize_shape, self.resize_shape))
                    for dicom_slice in cropped_dicom
                ]
            ),
            axis=-1,
        )
        breakpoint()

    @staticmethod
    def perform_preprocess(image_volume: np.ndarray):
        """

        :param image_volume:
        :return:
        """
        pre = ImagePreProcess(image_volume)
        preprocessed_image = pre.apply_preprocess()
        return preprocessed_image.reshape(image_volume.shape)

    @staticmethod
    def crop_slice(arr, mask):
        """
        
        :param arr:
        :param mask:
        :return:
        """
        return arr[
            tuple(
                slice(np.min(indexes), np.max(indexes) + 1)
                for indexes in np.where(mask)
            )
        ]
