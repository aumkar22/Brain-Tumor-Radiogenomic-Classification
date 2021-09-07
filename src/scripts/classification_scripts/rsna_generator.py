import math
import random
import tensorflow as tf
import tensorflow_io as tfio
import pandas as pd

from typing import NoReturn
from pathlib import Path
from tensorflow.keras.utils import Sequence


class RsnaDataGenerator(Sequence):
    def __init__(
        self,
        data: pd.DataFrame,
        batch_size: int,
        dicom_path: Path,
        shuffle: bool = True,
        train: bool = False,
    ):

        self.batch_size = batch_size
        self.data = data
        self.dicom_path = dicom_path
        self.patient_ids = self.data["BraTS21ID"].str.zfill(5).values
        self.label = self.data["MGMT_value"].values
        self.shuffle = shuffle
        self.train = train
        self.modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
        self._shuffle_indices(self.shuffle)

    def __len__(self) -> int:
        """
        Calculate the number of batches that can be created using the provided batch size.
        :return: Number of batches.
        """
        return int(math.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, index: int):

        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_indices = self.patient_ids[start_index:end_index]

        # feature_batch =

    def on_epoch_end(self) -> NoReturn:
        """
        Method that is called once training has finished an epoch. The only thing we need to do in
        those situations is shuffling the indices if that's been enabled in the constructor.
        :return: No return.
        """
        super().on_epoch_end()
        self._shuffle_indices(self.shuffle)

    def _shuffle_indices(self, shuffle) -> NoReturn:
        """
        Shuffle indices if shuffle is True. Shuffle has been added as an explicit parameter instead
        of just using self.shuffle to provide more intuition from callsites that shuffling is
        optional.
        :param shuffle: Whether or not data should be shuffled.
        :return: No return. Data is shuffled inplace.
        """
        if shuffle:
            random.shuffle(self.indices)

    def load_dicom(self, patient_indices: str):

        dicom_dataset = tf.data.Dataset.list_files(
            str(self.dicom_path) + f"/{patient_indices}/"
        )
        dicom_bytes = tf.io.read_file()
