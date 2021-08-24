from pathlib import Path

PROJECT_PATH: Path = Path(__file__).parent.parent.parent

DATA_FOLDER: Path = PROJECT_PATH / "data"
TRAIN_FOLDER: Path = DATA_FOLDER / "train"
TEST_FOLDER: Path = DATA_FOLDER / "test"

PREPROCESSED_PATH: Path = DATA_FOLDER / "preprocessed"
TRAIN_LABELS: Path = DATA_FOLDER / "train_labels.csv"
SUBMISSION_PATH: Path = DATA_FOLDER / "sample_submission.csv"
