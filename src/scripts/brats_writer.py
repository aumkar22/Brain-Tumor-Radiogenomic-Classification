import csv

from typing import NoReturn

from src.util.definitions import *
from src.scripts.brats_load import BratsLoadSave


def read_csv(csv_path: Path, train: bool = False) -> NoReturn:

    """
    Read data csv and run nifti to tfrecords conversion

    :param csv_path: Path to the csv
    :param train: True if reading training csv, else False
    """
    with open(str(csv_path), "r") as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader, None)
        if train:
            row_num = 5
        else:
            row_num = 4
        for row in reader:
            brats = BratsLoadSave(
                Path(csv_path.parent / row[row_num]), row[row_num], train
            )
            brats.nifti_to_tfrecords()


if __name__ == "__main__":

    train_csv = Path(BRATS_TRAIN_FOLDER / "name_mapping.csv")
    validation_csv = Path(BRATS_VALIDATION_FOLDER / "name_mapping_validation_data.csv")

    # read_csv(train_csv, True)
    read_csv(validation_csv)
