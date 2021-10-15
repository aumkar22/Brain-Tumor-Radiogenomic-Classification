import yaml
import argparse
import numpy as np

from typing import NoReturn

from src.util.definitions import *
from src.scripts.classification_scripts.rsna_generator import RsnaDataGenerator
from src.scripts.classification_scripts.models.base_model import ResNet
from src.scripts.classification_scripts.models.tl_model import TlModel
from src.scripts.classification_scripts.models.nn_models import NnModel
from src.scripts.classification_scripts.models.eval import plot_history


def train(model: NnModel, save_path: Path, model_name: str) -> NoReturn:
    """
    Function to train the model

    :param model: Baseline or transfer learning model
    :param save_path: Path to save the predictions
    :param model_name: Name of the model to train
    """

    compiled_model, callbacks = model.model_compile(print_summary=True)
    train_files = glob(str(TRAIN_NUMPY_FOLDER) + "/*.npz")
    train_generator = RsnaDataGenerator(data_path=train_files, batch_size=1, train=True)
    validation_files = glob(str(VALIDATION_NUMPY_FILES) + "/*.npz")
    validation_generator = RsnaDataGenerator(
        data_path=validation_files, batch_size=1, train=True
    )
    test_files = glob(str(TEST_NUMPY_FILES) + "/*.npz")
    test_generator = RsnaDataGenerator(data_path=test_files, batch_size=1, train=False)
    print("Training...")
    history = compiled_model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=200,
        callbacks=callbacks,
    )
    plot_history(history, save_path, model_name, True)

    print("Predicting on test set")
    test_predict = compiled_model.predict(
        test_generator, use_multiprocessing=True, workers=6
    )
    predictions = np.argmax(test_predict, 1)
    np.savez_compressed(str(save_path / (model_name + ".npz")), predictions=predictions)


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

    train(get_model, model_save_path, model_)
    # breakpoint()
