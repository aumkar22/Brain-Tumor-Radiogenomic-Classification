import pandas as pd
import csv

from typing import NoReturn

from src.scripts.classification_scripts.rsna_load_new import RsnaLoad
from src.util.definitions import *


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

    read_rsna_csv(TRAIN_LABELS)
