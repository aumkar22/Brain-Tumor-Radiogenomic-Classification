import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.util.definitions import BRATS_TRAIN_FOLDER, BRATS_VALIDATION_FOLDER
from src.util.folder_check import path_check
from src.scripts.data_preprocess import ImagePreProcess
from src.util.type_conversions import sitk_to_numpy


class BratsLoadSave(object):
    def __init__(self, data_path: Path, patient: str, preprocess: bool):

        """

        :param data_path:
        :param patient:
        :param preprocess:
        """

        self.data_path = data_path
        self.preprocess = preprocess
        self.patient = patient
        self.patient_flair = f"{self.patient}_flair.nii"
        self.patient_t1 = f"{self.patient}_t1.nii"
        self.patient_t1ce = f"{self.patient}_t1ce.nii"
        self.patient_t2 = f"{self.patient}_t2.nii"
        self.patient_mask = f"{self.patient}_seg.nii"

    @staticmethod
    def load_brats_nifti(nifti_data: str, preprocess: bool = True) -> np.ndarray:

        """

        :param nifti_data:
        :param preprocess:
        :return:
        """

        loaded_image = sitk.ReadImage(nifti_data)

        if preprocess:
            pre = ImagePreProcess(loaded_image)
            preprocessed_image = pre.apply_preprocess()
            return preprocessed_image

        return sitk_to_numpy(loaded_image)

    def load_preprocess(self):

        """

        :return:
        """

        path_check(self.data_path)
