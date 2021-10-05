import yaml
import argparse
import tensorflow as tf

from typing import List
from tensorflow.keras.callbacks import Callback

from src.util.definitions import *
from src.scripts.classification_scripts.rsna_generator import RsnaDataGenerator
from src.scripts.classification_scripts.models.base_model import ResNet


def train(model: tf.keras.Model, callbacks: List[Callback]):

    train_files = glob(str(TRAIN_NUMPY_FOLDER) + "/*.npz")
    train_generator = RsnaDataGenerator(data_path=train_files, batch_size=2, train=True)
    validation_files = glob(str(VALIDATION_NUMPY_FILES) + "/*.npz")
    validation_generator = RsnaDataGenerator(
        data_path=validation_files, batch_size=2, train=True
    )
    test_files = glob(str(TEST_NUMPY_FILES) + "/*.npz")
    test_generator = RsnaDataGenerator(data_path=test_files, batch_size=2, train=False)
    breakpoint()
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=200,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=True,
    )


if __name__ == "__main__":

    resnet_params_path = Path(CONFIG_FOLDER / "resnet_params.yaml")
    resnet_params = yaml.safe_load(resnet_params_path.open())["default"]
    model_save_path = Path(DATA_FOLDER / "models" / "resnet")
    resnet_params.update({"save_path": model_save_path})
    resnet_model = ResNet(**resnet_params)
    compiled_model, model_callbacks = resnet_model.model_compile(print_summary=True)
    train(compiled_model, model_callbacks)
    breakpoint()
