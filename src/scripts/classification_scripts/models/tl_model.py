import tensorflow.keras as tf
import efficientnet_3D.tfkeras as efn

from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling3D, Dense
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

        input_tensor = Input(input_shape=self.input_shape)

        if self.tl_model == "efficientnet":
            extracted_features = efn.EfficientNetB7(
                input_shape=self.input_shape, include_top=False, weights="imagenet"
            )
        else:
            tl_model_, preprocess_input = Classifiers.get(self.tl_model)
            extracted_features = tl_model_(
                input_shape=self.input_shape, include_top=False, weights="imagenet"
            )

        extracted_features_out = extracted_features(input_tensor)

        pool = GlobalAveragePooling3D()(extracted_features_out)

        out = Dense(self.out, activation="sigmoid")(pool)

        return Model(inputs=input_tensor, outputs=out)
