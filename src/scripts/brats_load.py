import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Tuple

from src.util.definitions import BRATS_TRAIN_FOLDER, BRATS_VALIDATION_FOLDER
from src.util.folder_check import path_check
from src.scripts.data_preprocess import ImagePreProcess
from src.util.type_conversions import sitk_to_numpy


class BratsLoadSave(object):
    def __init__(self, data_path: Path, patient: str):

        """
        Intialize data parameters

        :param data_path: Path to nifti scans for every patient
        :param patient: Patient ID
        """

        self.data_path = data_path
        self.patient = patient
        self.patient_flair = f"{self.patient}_flair.nii"
        self.patient_t1 = f"{self.patient}_t1.nii"
        self.patient_t1ce = f"{self.patient}_t1ce.nii"
        self.patient_t2 = f"{self.patient}_t2.nii"
        self.patient_mask = f"{self.patient}_seg.nii"

    @staticmethod
    def load_brats_nifti(nifti_data: str, preprocess: bool = True) -> np.ndarray:

        """
        Function to load nifti images and preprocess them

        :param nifti_data: Nifti scans
        :param preprocess: True for MRI scans, False for mask
        :return: Preprocessed scans
        """

        loaded_image = sitk.ReadImage(nifti_data)

        if preprocess:
            pre = ImagePreProcess(loaded_image)
            preprocessed_image = pre.apply_preprocess()
            return preprocessed_image

        return sitk_to_numpy(loaded_image)

    def load_preprocess(self) -> Tuple[np.ndarray, np.ndarray]:

        """
        Function to create stacked volumes of different MRI scans

        :return: Tuple of preprocessed scans and mask (without preprocessing)
        """

        path_check(self.data_path)
        flair_array = self.load_brats_nifti(
            str(Path(self.data_path / self.patient_flair))
        )
        t1_array = self.load_brats_nifti(str(Path(self.data_path / self.patient_t1)))
        t1ce_array = self.load_brats_nifti(
            str(Path(self.data_path / self.patient_t1ce))
        )
        t2_array = self.load_brats_nifti(str(Path(self.data_path / self.patient_t2)))
        mask_array = self.load_brats_nifti(
            str(Path(self.data_path / self.patient_mask)), False
        )

        return np.stack((flair_array, t1_array, t1ce_array, t2_array)), mask_array

    @staticmethod
    def __int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def __bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __serialize(self, scans: np.ndarray, masks: np.ndarray):

        """

        :param scans: 
        :param masks:
        :return:
        """

        writer = tf.io.TFRecordWriter(str(self.data_path.stem) + ".tfrecords")
        scans_raw = scans.tostring()
        masks_raw = masks.tostring()

        features = tf.train.Features(
            feature={
                "patient_id": self.__int64_feature(self.patient),
                "scan": self.__bytes_feature(scans_raw),
                "mask": self.__bytes_feature(masks_raw),
            }
        )
        feature_example = tf.train.Example(features=features)
        writer.write(feature_example.SerializeToString())
        writer.close()

    def nifti_to_tfrecords(self):

        mri_scans, mask = self.load_preprocess()
        self.__serialize(mri_scans, mask)
