import csv

from typing import NoReturn

from src.util.definitions import *
from src.scripts.brats_load import BratsLoadSave


def read_csv(csv_path: Path) -> NoReturn:

    """
    Read data csv and run nifti to tfrecords conversion

    :param csv_path: Path to the csv
    """
    with open(str(csv_path), "r") as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader, None)
        for row in reader:
            brats = BratsLoadSave(Path(csv_path.parent / row[5]), row[5])
            brats.nifti_to_tfrecords()
            breakpoint()


if __name__ == "__main__":

    train_csv = Path(BRATS_TRAIN_FOLDER / "name_mapping.csv")
    validation_csv = Path(BRATS_VALIDATION_FOLDER / "name_mapping_validation_data.csv")

    read_csv(train_csv)
    read_csv(validation_csv)
