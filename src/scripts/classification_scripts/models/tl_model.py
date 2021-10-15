import tensorflow.keras as tf
import efficientnet_3D.tfkeras as efn

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling3D, Dense, Conv3D
from classification_models_3D.tfkeras import Classifiers
from pathlib import Path

from src.scripts.classification_scripts.models.nn_models import NnModel


class TlModel(NnModel):
    """
    Class for transfer learning models
    """

    def __init__(self, save_path: Path, tl_model: str):

        super().__init__(save_path)
        self.tl_model = tl_model

    def model_architecture(self) -> tf.Model:
        """
        Transfer learning models

        :return:
        """

        input_tensor = Input(shape=self.input_shape)
        mapping3feat = Conv3D(
            3, (3, 3, 3), strides=(1, 1, 1), padding="same", use_bias=True
        )(input_tensor)

        if self.tl_model == "efficientnet":
            extracted_features = efn.EfficientNetB7(
                input_shape=(30, 240, 240, 3), include_top=False, weights="imagenet"
            )
        else:
            tl_model_, preprocess_input = Classifiers.get(self.tl_model)
            extracted_features = tl_model_(
                input_shape=(30, 240, 240, 3), include_top=False, weights="imagenet"
            )

        extracted_features_out = extracted_features(mapping3feat)

        pool = GlobalAveragePooling3D()(extracted_features_out)

        out = Dense(self.out, activation="sigmoid")(pool)

        return Model(inputs=input_tensor, outputs=out)
