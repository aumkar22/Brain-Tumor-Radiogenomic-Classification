import pydicom as dicom
import numpy as np
import cv2

from pathlib import Path

# from src.preprocessing.rsna_preprocess import *
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

    def dicom_load(self, dicom_file: str):

        dicom_data = dicom.read_file(dicom_file).pixel_array
        breakpoint()
        resized_dicom = cv2.resize(
            dicom_data, (self.resize_shape, self.resize_shape, 1)
        )

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

        dicom_volume = np.stack(dicom_data)
