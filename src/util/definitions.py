from pathlib import Path
from glob import glob

PROJECT_PATH: Path = Path(__file__).parent.parent.parent

DATA_FOLDER: Path = PROJECT_PATH / "data"
TRAIN_FOLDER: Path = DATA_FOLDER / "train"
TRAIN_NUMPY_FOLDER: Path = DATA_FOLDER / "train_npy"
VALIDATION_NUMPY_FILES: Path = DATA_FOLDER / "validation_npy"
TEST_NUMPY_FILES: Path = DATA_FOLDER / "test_npy"
TRAIN_FILES = glob(str(TRAIN_FOLDER) + "/*/*/*.dcm")
CONFIG_FOLDER: Path = PROJECT_PATH / "src" / "util" / "config"
VALIDATION_FOLDER: Path = DATA_FOLDER / "test"
VALIDATION_FILES = glob(str(VALIDATION_FOLDER) + "/*/*/*.dcm")
BRATS_TRAIN_FOLDER: Path = DATA_FOLDER / "brats_data" / "MICCAI_BraTS2020_TrainingData"
BRATS_VALIDATION_FOLDER: Path = DATA_FOLDER / "brats_data" / "MICCAI_BraTS2020_ValidationData"

PREPROCESSED_PATH: Path = DATA_FOLDER / "preprocessed"
TRAIN_LABELS: Path = DATA_FOLDER / "train_labels.csv"
SUBMISSION_PATH: Path = DATA_FOLDER / "sample_submission.csv"

classes = ["0", "1"]
