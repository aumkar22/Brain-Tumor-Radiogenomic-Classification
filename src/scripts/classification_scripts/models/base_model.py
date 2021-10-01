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

from src.scripts.classification_scripts.models.residual_layer import Residual


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
