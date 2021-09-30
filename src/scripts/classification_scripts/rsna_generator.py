import math
import numpy as np

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

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        """
        Create and return a batch of data and corresponding labels.

        :param index: Batch index
        :return: A tuple consisting of a batch of feature data and labels. During testing,
        only feature batch data is returned
        """

        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_path = self.data_path[start_index:end_index]

        feature_batch = np.array(
            [
                np.expand_dims(np.load(feature)["flair_volume"], axis=-1)
                for feature in batch_path
            ]
        )
        if self.train:
            label_batch = np.array(
                [np.load(feature)["label"] for feature in batch_path]
            )
            return feature_batch, label_batch
        else:
            return feature_batch
