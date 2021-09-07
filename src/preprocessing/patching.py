import numpy as np

from typing import Tuple


class Patching:

    """
    Base code taken from: https://github.com/LauraMoraB/BrainTumorSegmentation/blob/master/src/dataset/patching/random_tumor_distribution.py
    """

    def __init__(self, mask: np.ndarray, scans: np.ndarray, patch_size: Tuple):

        self.mask = mask
        self.scans = scans
        self.patch_size = patch_size


    def check_if_all_no_tumour(self, patch_mask: np.ndarray):

        if np.all(patch_mask == 0):




    def random_center_tumor(self):

        tumor_indices = np.nonzero(self.mask)
        start_x = np.random.randint(0, tumor_indices[0])
        start_y = np.random.randint(0, tumor_indices[1])
        start_z = np.random.randint(0, tumor_indices[2])

        center_coord = (
            tumor_indices[0][start_x],
            tumor_indices[1][start_y],
            tumor_indices[2][start_z],
        )
