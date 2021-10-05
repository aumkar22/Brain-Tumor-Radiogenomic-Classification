import tensorflow.keras as tf

from tensorflow.keras.layers import (
    Input,
    Conv3D,
    MaxPool3D,
    BatchNormalization,
    GlobalAveragePooling3D,
    Dropout,
    Dense,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import Callback
from typing import List, Tuple

from src.scripts.classification_scripts.models.residual_layer import Residual
from src.scripts.classification_scripts.models.callbacks import get_callbacks


class ResNet:
    def __init__(
        self,
        N1,
        N2,
        N3,
        kernel_size1,
        pool_size1,
        kernel_size2,
        pool_size2,
        kernel_size3,
        pool_size3,
        dilation_rate1,
        dilation_rate2,
        dilation_rate3,
        dilation_rate4,
        Nfc1,
        Nfc2,
        dropout1,
        dropout2,
        save_path,
        out=1,
        input_shape=(30, 240, 240, 1),
    ):

        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.kernel_size1 = kernel_size1
        self.pool_size1 = pool_size1
        self.kernel_size2 = kernel_size2
        self.pool_size2 = pool_size2
        self.kernel_size3 = kernel_size3
        self.pool_size3 = pool_size3
        self.Nfc1 = Nfc1
        self.Nfc2 = Nfc2
        self.dilation_rate1 = dilation_rate1
        self.dilation_rate2 = dilation_rate2
        self.dilation_rate3 = dilation_rate3
        self.dilation_rate4 = dilation_rate4
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.out = out
        self.input_shape = input_shape
        self.save_path = save_path

    def model(self) -> tf.Model:

        model_input = Input(shape=self.input_shape)

        model = Conv3D(
            self.N1, kernel_size=self.kernel_size1, activation="relu", padding="same",
        )(model_input)

        model = BatchNormalization(axis=-1, scale=None)(model)

        model = Residual(self.N1, self.kernel_size1, self.dilation_rate1)(model)

        model = Residual(self.N1, self.kernel_size1, self.dilation_rate2)(model)

        model = Residual(self.N1, self.kernel_size1, self.dilation_rate3)(model)

        model = Residual(self.N1, self.kernel_size1, self.dilation_rate4)(model)

        model = Conv3D(self.N2, kernel_size=self.kernel_size2, activation="relu")(model)

        model = MaxPool3D(self.pool_size1)(model)

        model = BatchNormalization(axis=-1, scale=None)(model)

        model = Conv3D(self.N3, kernel_size=self.kernel_size3, activation="relu")(model)

        model = MaxPool3D(self.pool_size2)(model)

        model = BatchNormalization(axis=-1, scale=None)(model)

        model = GlobalAveragePooling3D()(model)

        model = Dense(self.Nfc1, activation="relu")(model)

        model = Dropout(self.dropout1)(model)

        model = Dense(self.Nfc2, activation="relu")(model)

        model = Dropout(self.dropout2)(model)

        out = Dense(self.out, activation="sigmoid")(model)

        return Model(inputs=[model_input], outputs=out)

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

        resnet_model = self.model()
        early_stopping, model_checkpoint, step_decay_lr = get_callbacks(self.save_path)
        adam = Adam(lr=step_decay_lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
        auc = AUC(name="roc_auc")
        resnet_model.compile(
            loss="binary_crossentropy", optimizer=adam, metrics=["accuracy", auc]
        )
        if print_summary:
            resnet_model.summary()

        return resnet_model, [early_stopping, model_checkpoint]
