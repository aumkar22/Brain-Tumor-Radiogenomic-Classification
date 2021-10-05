from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
)
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import Callback
from pathlib import Path
from typing import Tuple

from src.util.folder_check import path_check


def get_callbacks(
    save_path: Path, initial_learning_rate: float = 1e-3
) -> Tuple[Callback, Callback, ExponentialDecay]:
    """
    Function to build model callbacks

    :param save_path: Model checkpoint save path
    :param initial_learning_rate: Initial learning rate for exponential decay
    :return: Model callbacks
    """

    path_check(save_path, True)
    early_stopping = EarlyStopping(monitor="val_roc_auc", patience=3, mode="max")
    model_checkpoint = ModelCheckpoint(
        filepath=save_path, monitor="val_roc_auc", save_best_only=True, mode="max"
    )
    step_decay_lr = ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    return early_stopping, model_checkpoint, step_decay_lr
