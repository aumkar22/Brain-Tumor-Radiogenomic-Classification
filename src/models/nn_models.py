import tensorflow.keras as tf

from pathlib import Path
from abc import ABC, abstractmethod
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
)
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from typing import Tuple, List

from src.util.folder_check import path_check


class NnModel(ABC):
    """
    Generic neural network model class. Any deep learning architecture
    in this project should inherit this class for a consistent interface
    """

    def __init__(
        self, save_path: Path, input_shape: Tuple = (30, 240, 240, 1), out: int = 1
    ):
        """

        :param input_shape:
        :param out:
        """

        self.save_path = save_path
        self.input_shape = input_shape
        self.out = out

    @abstractmethod
    def model_architecture(self) -> tf.Model:
        """"
        Function to build model architecture
        """
        pass

    @staticmethod
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
        early_stopping = EarlyStopping(monitor="val_roc_auc", patience=5, mode="max")
        model_checkpoint = ModelCheckpoint(
            filepath=save_path, monitor="val_roc_auc", save_best_only=True, mode="max"
        )
        step_decay_lr = ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )

        return early_stopping, model_checkpoint, step_decay_lr

    def model_compile(
        self,
        print_summary: bool = False,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> Tuple[tf.Model, List[Callback]]:
        """
        Function to compile the model

        :param print_summary: Print model summary if true
        :param beta1: Exponential decay rate for the running average of the gradient
        :param beta2: Exponential decay rate for the running average of the square of the gradient
        :param epsilon: Epsilon parameter to prevent division by zero error
        :return: Compiled Keras model, list of callbacks
        """

        model_architecture = self.model_architecture()
        early_stopping, model_checkpoint, step_decay_lr = self.get_callbacks(
            self.save_path
        )
        adam = Adam(
            learning_rate=step_decay_lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon
        )
        auc = AUC(name="roc_auc")
        model_architecture.compile(
            loss="binary_crossentropy", optimizer=adam, metrics=["accuracy", auc]
        )
        if print_summary:
            model_architecture.summary()

        return model_architecture, [early_stopping, model_checkpoint]
