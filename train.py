import yaml
import argparse
import numpy as np

from typing import NoReturn

from src.util.definitions import *
from src.scripts.classification_scripts.rsna_generator import RsnaDataGenerator
from src.scripts.classification_scripts.models.base_model import ResNet
from src.scripts.classification_scripts.models.tl_model import TlModel
from src.scripts.classification_scripts.models.nn_models import NnModel


def train(model: NnModel) -> NoReturn:

    compiled_model, callbacks = model.model_compile(print_summary=True)
    train_files = glob(str(TRAIN_NUMPY_FOLDER) + "/*.npz")
    train_generator = RsnaDataGenerator(data_path=train_files, batch_size=1, train=True)
    validation_files = glob(str(VALIDATION_NUMPY_FILES) + "/*.npz")
    validation_generator = RsnaDataGenerator(
        data_path=validation_files, batch_size=1, train=True
    )
    validation_generator_for_evaluation = RsnaDataGenerator(
        data_path=validation_files, batch_size=32, train=False
    )
    test_files = glob(str(TEST_NUMPY_FILES) + "/*.npz")
    test_generator = RsnaDataGenerator(data_path=test_files, batch_size=1, train=False)
    breakpoint()
    print("Training...")
    compiled_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=200,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=True,
    )

    print("Predicting on test set")
    val_predict = compiled_model.predict(
        validation_generator_for_evaluation, use_multiprocessing=True, workers=6
    )
    predictions = np.argmax(val_predict, 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="Select model to train",
        choices=[
            "baseline",
            "resnet101",
            "resnet152",
            "seresnet50",
            "seresnet101",
            "seresnet152",
            "seresnext50",
            "seresnext101",
            "senet154",
            "resnext50",
            "resnext101",
            "vgg16",
            "vgg19",
            "inceptionresnetv2",
            "inceptionv3",
            "efficientnet",
        ],
        default="baseline",
    )

    args = parser.parse_args()
    model_ = args.model

    if model_ == "baseline":
        module_name, model_class = "base_model", "ResNet"
        resnet_params_path = Path(CONFIG_FOLDER / "resnet_params.yaml")
        resnet_params = yaml.safe_load(resnet_params_path.open())["default"]
        model_save_path = Path(DATA_FOLDER / "models" / "resnet")
        resnet_params.update({"save_path": model_save_path})
        get_model = ResNet(**resnet_params)
    else:
        module_name, model_class = "tl_model", "TlModel"
        model_save_path = Path(DATA_FOLDER / "models" / model_)
        get_model = TlModel(model_save_path, model_)

    train(get_model)
    breakpoint()
