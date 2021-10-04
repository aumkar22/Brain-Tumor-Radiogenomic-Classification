import math
import numpy as np
import tensorflow as tf

from typing import Union, Tuple, List
from tensorflow.keras.utils import Sequence


class RsnaDataGenerator(Sequence):
    """
    RSNA batch generator class
    """

    def __init__(
        self, data_path: List[str], batch_size: int, train: bool = False,
    ):

        """
        :param data_path: List of paths to numpy preprocessed data
        :param batch_size: Batch size
        :param train: True for train/validation, False for test
        """

        self.data_path = data_path
        self.batch_size = batch_size
        self.train = train

    def __len__(self) -> int:
        """
        Calculate the number of batches that can be created using the provided batch size.

        :return: Number of batches.
        """
        return int(math.ceil(len(self.data_path) / self.batch_size))

    def __getitem__(self, index: int) -> Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]:

        """
        Create and return a batch of data and corresponding labels.

        :param index: Batch index
        :return: A tuple consisting of a batch of feature data and labels. During testing,
        only feature batch data is returned
        """

        if self.train:
            feature_dataset = tf.data.Dataset.from_tensor_slices(self.data_path).map(
                lambda x: tf.py_function(
                    func=self.load_numpy_volumes, inp=[x], Tout=[tf.float32, tf.float32]
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
        else:
            feature_dataset = tf.data.Dataset.from_tensor_slices(self.data_path).map(
                lambda x: tf.py_function(
                    func=self.load_numpy_volumes, inp=[x], Tout=[tf.float32]
                ),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )

        feature_dataset = feature_dataset.cache()
        feature_dataset = feature_dataset.batch(self.batch_size)
        feature_dataset = feature_dataset.shuffle(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        feature_dataset = feature_dataset.repeat()
        feature_dataset = feature_dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        if self.train:
            feature_batch, feature_label = next(iter(feature_dataset))
            return feature_batch, feature_label
        else:
            feature_batch = next(iter(feature_dataset))
            return feature_batch

    def load_numpy_volumes(
        self, feature_path: str
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Read numpy files

        :param feature_path: Path to numpy files
        :return: Returns feature data array and labels for training/validation and only feature
        data array for test
        """

        feature_data = np.expand_dims(np.load(feature_path)["flair_volume"], axis=-1)

        if self.train:
            label_data = np.load(feature_path)["label"]
            return feature_data, label_data

        return feature_data
