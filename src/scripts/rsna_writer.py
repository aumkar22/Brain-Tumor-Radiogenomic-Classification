import pandas as pd
import csv

from typing import NoReturn, Tuple
from sklearn.model_selection import GroupShuffleSplit

from src.scripts.rsna_load import RsnaLoad
from src.util.definitions import *
from src.util.folder_check import path_check


def to_csv(data: pd.DataFrame, save_path: Path, train: bool = False) -> NoReturn:

    """
    Save train/validation dataframe to csv

    :param data: Dataframe which has patient ID and corresponding label
    :param save_path: Path to save the csv
    :param train: True, if train dataframe, else False
    """

    if train:
        csv_save_path = Path(save_path / "train_npy")
    else:
        csv_save_path = Path(save_path / "validation_npy")

    path_check(csv_save_path, True)
    data.to_csv(str(Path(csv_save_path / "data_labels.csv")), index=False)


def create_splits(rsna_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Split labels csv between train and validation based on patient ID as group

    :param rsna_csv: Path to train labels csv
    :return: Tuple of training and validation dataframes
    """

    labels_csv = pd.read_csv(str(rsna_csv))
    train_indices, val_indices = next(
        GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=42).split(
            labels_csv, groups=labels_csv["BraTS21ID"]
        )
    )
    return labels_csv.iloc[train_indices], labels_csv.iloc[val_indices]


def read_rsna_csv(rsna_csv: Path, train: bool) -> NoReturn:

    """
    Function to read csv with patient ID and corresponding label and save preprocessed data

    :param rsna_csv: Path to the csv file
    :param train: True for train/validation, False for test
    """

    with open(str(rsna_csv), "r") as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader, None)
        for row in reader:
            if train:
                rsna = RsnaLoad(
                    rsna_csv.parent, row[0].zfill(5), int(row[1]), train=train
                )
            else:
                rsna = RsnaLoad(rsna_csv.parent, row[0].zfill(5), None, train=train)
            rsna.save_npy_volume()


if __name__ == "__main__":

    train_df, val_df = create_splits(TRAIN_LABELS)
    to_csv(train_df, DATA_FOLDER, True)
    to_csv(val_df, DATA_FOLDER)
    read_rsna_csv(Path(DATA_FOLDER / "train_npy" / "data_labels.csv"), True)
    read_rsna_csv(Path(DATA_FOLDER / "validation_npy" / "data_labels.csv"), True)
    path_check(Path(DATA_FOLDER / "test_npy"), True)
    read_rsna_csv(Path(DATA_FOLDER / "sample_submission.csv"), False)
