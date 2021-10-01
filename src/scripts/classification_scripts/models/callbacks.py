from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
)
from pathlib import Path
from typing import Tuple

from src.util.folder_check import path_check


def step_decay(epoch, learning_rate) -> float:
    """
    Drop learning rate every 10 epochs

    :param epoch: Epoch index
    :param learning_rate: Learning rate at every epoch index
    :return: Dropped learning rate
    """

    drop = 0.4
    epochs_drop = 10.0
    new_lr = learning_rate * (drop ** ((1 + epoch) // epochs_drop))

    if new_lr < 4e-5:
        new_lr = 4e-5

    print(f"Changing learning rate to {new_lr}")

    return new_lr


def get_callbacks(save_path: Path) -> Tuple:
    """
    Function to build model callbacks

    :param save_path: Model checkpoint save path
    :return: Model callbacks
    """

    path_check(save_path, True)
    early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    model_checkpoint = ModelCheckpoint(
        filepath=save_path, monitor="val_loss", save_best_only=True
    )
    step_decay_lr = LearningRateScheduler(step_decay)

    return early_stopping, model_checkpoint, step_decay_lr
