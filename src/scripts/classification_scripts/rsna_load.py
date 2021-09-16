import tensorflow_io as tfio
import numpy as np

from pathlib import Path

from src.preprocessing.rsna_preprocess import *
from src.scripts.data_load import *


class RsnaLoad(DataLoad):
    def __init__(self, data_path: Path, patient: str, label: int):
        self.data_path = data_path
        self.patient = patient
        self.label = label

    @staticmethod
    def dicom_read(dicom_file: str) -> np.ndarray:
        """
        Function to read one dicom file and convert to a numpy array

        :param dicom_file: Path to dicom file
        :return: Numpy array converted from dicom file
        """
        dicom_bytes = tf.io.read_file(dicom_file)
        dicom_decoded = tfio.image.decode_dicom_image(dicom_bytes, dtype=tf.uint16)

        return dicom_decoded.numpy()

    def perform_preprocess(self, image_volume: tf.Tensor) -> tf.Tensor:
        """

        :return:
        """

        contrast_enhanced_image = tf_equalize_histogram(image_volume)
        normalized_image = normalization(contrast_enhanced_image)

        return tf.reshape(normalized_image, tf.shape(image_volume))

    def data_load(self):

        """

        :return:
        """

        dicom_path = Path(self.data_path / f"{self.patient}" / "FLAIR")
        dicom_list = [str(i) for i in dicom_path.glob(r"**/*.dcm")]
        dicom_data = []
        for dicom_index, dicom_file in enumerate(dicom_list):
            dicom_data.append(self.dicom_read(dicom_file))

        dicom_volume = np.stack(dicom_data)

        tf_dataset = tf.data.Dataset.from_tensor_slices(dicom_volume)
        tf_dataset = tf_dataset.map(
            self.perform_preprocess, num_parallel_calls=tf.data.AUTOTUNE
        )

    def __serialize(self):

        writer = tf.io.TFRecordWriter(
            str(Path(self.data_path / self.data_path.stem)) + ".tfrecords"
        )
        features = tf.train.Features(
            feature={
                "patient_id": self.__int64_feature(int(self.patient[-3:])),
                "scan": self.__bytes_feature(scans_raw),
                "mask": self.__bytes_feature(masks_raw),
            }
        )

    def data_to_tfrecords(self):

        writer = tf.io.TFRecordWriter(
            str(Path(self.data_path / self.data_path.stem)) + ".tfrecords"
        )
