import pandas as pd
import csv

from typing import NoReturn
from sklearn.model_selection import GroupShuffleSplit

from src.scripts.classification_scripts.rsna_load_new import RsnaLoad
from src.util.definitions import *


def create_splits(rsna_csv: Path):

    labels_csv = pd.read_csv(str(rsna_csv))
    train_indices, val_indices = next(
        GroupShuffleSplit(test_size=0.2, n_splits=2).split(
            labels_csv, groups=labels_csv["BraTS21ID"]
        )
    )
    breakpoint()
    return labels_csv.iloc[train_indices], labels_csv.iloc[val_indices]


def read_rsna_csv(rsna_csv: Path) -> NoReturn:

    """

    :param rsna_csv:
    """

    with open(str(rsna_csv), "r") as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader, None)
        for row in reader:
            rsna = RsnaLoad(TRAIN_FOLDER, row[0].zfill(5), int(row[1]))
            rsna.data_load()


if __name__ == "__main__":

    train, val = create_splits(TRAIN_LABELS)
    breakpoint()
    read_rsna_csv(TRAIN_LABELS)
