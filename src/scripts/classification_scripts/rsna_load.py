import tensorflow_io as tfio
import numpy as np

from pathlib import Path

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

        contrast_enhanced_image = self.tf_equalize_histogram(image_volume)
        normalized_image = self.normalization(contrast_enhanced_image)

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

    @tf.function
    def normalization(self, image_volume: tf.Tensor) -> tf.Tensor:

        """

        :param image_volume:
        :return:
        """

        normalized_image = (
            image_volume - tf.reduce_mean(image_volume)
        ) / tf.math.reduce_std(image_volume)

        return normalized_image

    @tf.function
    def tf_equalize_histogram(self, image_volume: tf.Tensor) -> tf.Tensor:

        """
        Contrast enhancement using histogram equalization. Code taken from:
        https://stackoverflow.com/questions/42835247/how-to-implement-histogram-equalization-for-images-in-tensorflow

        :param image_volume:
        :return:
        """

        values_range = tf.constant([0.0, 255.0], dtype=tf.float32)
        histogram = tf.histogram_fixed_width(
            tf.compat.v1.to_float(image_volume), values_range, 256
        )
        cdf = tf.cumsum(histogram)
        cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

        img_shape = tf.shape(image_volume)
        pix_cnt = img_shape[-3] * img_shape[-2]
        px_map = tf.round(
            tf.compat.v1.to_float(cdf - cdf_min)
            * 255.0
            / tf.compat.v1.to_float(pix_cnt - 1)
        )
        px_map = tf.cast(px_map, tf.uint8)

        eq_hist = tf.expand_dims(
            tf.gather_nd(px_map, tf.cast(image_volume, tf.int32)), 2
        )
        return eq_hist

    def data_to_tfrecords(self):

        writer = tf.io.TFRecordWriter(
            str(Path(self.data_path / self.data_path.stem)) + ".tfrecords"
        )
