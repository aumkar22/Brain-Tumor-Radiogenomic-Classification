from tensorflow.keras.layers import Layer, Conv3D, BatchNormalization


class Residual(Layer):

    """
    Custom residual layer for ResNet model
    """

    def __init__(self, filters, kernel_size, dilation_rate):

        """
        Initializing layer parameters
        :param filters: Convolution filters
        :param kernel_size: Kernel size
        :param dilation_rate: Rate of dilation
        """

        super().__init__()
        self.conv3d = Conv3D(
            filters,
            kernel_size,
            padding="same",
            activation="relu",
            dilation_rate=dilation_rate,
        )
        self.batch_norm = BatchNormalization(axis=-1, scale=None)

    def call(self, input_tensor, training=False):

        """
        Performing dilated convolutions in the residual layer
        :param input_tensor: Input tensor
        :param training: Parameter for batch normalization
        :return: Dilated convolution residual layer output tensor
        """

        x = self.conv3d(input_tensor)
        x = self.batch_norm(x, training=training)

        x += input_tensor

        return x
