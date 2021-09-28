import pydicom as dicom
import cv2

from pathlib import Path
from typing import NoReturn

from src.preprocessing.data_preprocess import *


class RsnaLoad:

    """
    Data loading class for RSNA data
    """

    def __init__(
        self,
        data_path: Path,
        patient: str,
        label: int,
        num_images: int = 30,
        resize_shape: int = 240,
        train: bool = True,
    ):

        """
        Class initializations

        :param data_path: Path to dicom files
        :param patient: Patient ID
        :param label: Ground truth (MGMT value)
        :param num_images: Number of images to be selected for creating 3D volumes
        :param resize_shape: Resizing shape
        :param train: True for loading train/validation data, False for test data
        """
        self.data_path = data_path
        self.patient = patient
        self.train = train
        if self.train:
            self.label = label
        self.num_images = num_images
        self.resize_shape = resize_shape

    @staticmethod
    def dicom_load(dicom_file: str) -> np.ndarray:
        """
        Read dicom file

        :param dicom_file: Dicom file path
        :return: Dicom data as a numpy array
        """

        return dicom.read_file(dicom_file).pixel_array

    def data_load(self) -> np.ndarray:
        """
        Load and preprocess mri slices

        :return: Return cropped, preprocessed and stacked scan volumes
        """
        if self.train:
            dicom_path = Path(self.data_path.parent / "train" / self.patient / "FLAIR")
        else:
            dicom_path = Path(self.data_path.parent / "test" / self.patient / "FLAIR")

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

        resize_cropped_dicom = np.array(
            [
                cv2.resize(dicom_slice, (self.resize_shape, self.resize_shape))
                for dicom_slice in cropped_dicom
            ]
        )

        return resize_cropped_dicom

    def save_npy_volume(self) -> NoReturn:

        """
        Save scan volumes
        """

        dicom_volume = self.data_load()
        if self.train:
            np.savez_compressed(
                str(self.data_path / (self.patient + ".npz")),
                flair_volume=dicom_volume,
                label=np.array([float(self.label)]),
            )
        else:
            np.savez_compressed(
                str(self.data_path / (self.patient + ".npz")), flair_volume=dicom_volume
            )

    @staticmethod
    def perform_preprocess(image_volume: np.ndarray) -> np.ndarray:
        """
        Perform preprocessing on scan volumes

        :param image_volume: Scan volume
        :return: Preprocessed scan volume
        """
        pre = ImagePreProcess(image_volume)
        preprocessed_image = pre.apply_preprocess()
        return preprocessed_image.reshape(image_volume.shape)

    @staticmethod
    def crop_slice(scan_volume, mask) -> np.ndarray:
        """
        Crop out black edges from slices

        :param scan_volume: Scan volume
        :param mask: Non-black mask
        :return: Cropped scan volume
        """
        return scan_volume[
            tuple(
                slice(np.min(indexes), np.max(indexes) + 1)
                for indexes in np.where(mask)
            )
        ]
