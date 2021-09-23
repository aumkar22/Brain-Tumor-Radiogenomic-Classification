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
    def crop_dicom(dicom_array: np.ndarray) -> np.ndarray:
        """
        https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        :param dicom_array:
        :return:
        """
        # breakpoint()
        thresh = cv2.threshold(dicom_array, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        breakpoint()
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        return dicom_array[extTop[1] : extBot[1], extLeft[0] : extRight[0]]

    def dicom_load(self, dicom_file: str) -> np.ndarray:
        dicom_data = dicom.read_file(dicom_file).pixel_array
        resized_dicom = cv2.resize(dicom_data, (self.resize_shape, self.resize_shape))
        cropped_dicom = self.crop_dicom(cv2.convertScaleAbs(resized_dicom))
        return np.expand_dims(cropped_dicom, axis=-1)

    def data_load(self):
        dicom_path = Path(self.data_path / f"{self.patient}" / "FLAIR")
        dicom_list = [str(i) for i in dicom_path.glob(r"**/*.dcm")]

        middle_dicom = len(dicom_list) // 2
        divide_images = self.num_images // 2

        left_end = max(0, middle_dicom - divide_images)
        right_end = min(len(dicom_list), middle_dicom + divide_images)
        dicom_data = []
        for dicom_index, dicom_file in enumerate(dicom_list[left_end:right_end]):
            dicom_data.append(self.dicom_load(dicom_file))

        # breakpoint()
        dicom_volume = np.stack(dicom_data)
        preprocessed_dicom_volume = self.perform_preprocess(dicom_volume)
        breakpoint()

    @staticmethod
    def perform_preprocess(image_volume: np.ndarray):
        pre = ImagePreProcess(image_volume)
        preprocessed_image = pre.apply_preprocess()
        return preprocessed_image.reshape(image_volume.shape)
