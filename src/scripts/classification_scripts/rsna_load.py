from src.scripts.data_load import *


class RsnaLoad(DataLoad):
    def __init__(self, data_path: Path, patient: str):
        self.data_path = data_path
        self.patient = patient
