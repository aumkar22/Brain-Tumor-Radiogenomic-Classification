import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

from typing import NoReturn
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
from tensorflow.keras.callbacks import History

from src.util.definitions import classes
from src.util.folder_check import path_check


def plot_history(
    history: History, save_path: Path, model: str, plot_show: bool = False
) -> NoReturn:
    """
    Function to plot model performance on validation set

    :param history: History callback
    :param save_path: Path to save the plot
    :param model: Trained model name
    :param plot_show: Display plot if True
    """

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    fig.suptitle(f"{model} performance on validation set")

    ax1.plot(history.history["val_loss"])
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["val_accuracy"])
    ax2.set_ylabel("Accuracy")

    ax3.plot(history.history["val_roc_auc"])
    ax3.set_ylabel("AUC")
    ax3.set_xlabel("Epoch")

    if plot_show:
        plt.show()

    plt.savefig(str(save_path / (model + ".png")), dpi=300)


class EvalVisualize(object):

    """
    Class to perform model evaluation and plot confusion matrix.
    """

    def __init__(self, ytrue: np.ndarray, ypred: np.ndarray):

        """
        Initialize with ground truth and prediction arrays.
        :param ytrue: Ground truth array
        :param ypred: Predictions
        """

        self.ytrue = ytrue
        self.ypred = ypred

    def get_metrics(self, save_path: Path, print_report: bool = False) -> NoReturn:

        """
        Classification report returns precision, recall and F1 scores for each class.
        :param save_path: Path to save metrics.
        :param print_report: Boolean parameter. Print report if True.
        :return: No return.
        """

        result = classification_report(self.ytrue, self.ypred, target_names=classes)

        if print_report:
            print(result)

        path_check(save_path, True)
        pickle.dump(result, save_path.open("wb"))

    def get_confusion_matrix(
        self, save_path: Path, plot_matrix: bool = False
    ) -> NoReturn:

        """
        Function to plot confusion matrix.
        :param save_path: Path to save confusion matrix.
        :param plot_matrix: Boolean parameter. Plot if True.
        :return: No return.
        """

        cm = confusion_matrix(self.ytrue, self.ypred)
        normalized_cm = np.expand_dims((cm.astype("float") / cm.sum(axis=1)), axis=1)

        path_check(save_path, True)

        plt.figure(figsize=(25, 25))
        sns.heatmap(
            normalized_cm, annot=True, xticklabels=classes, yticklabels=classes, fmt="g"
        )
        plt.title("Normalized confusion matrix")
        plt.ylabel("True label", fontsize=30)
        plt.xlabel("Predicted label", fontsize=30)
        plt.savefig(save_path, dpi=400)

        if plot_matrix:
            plt.show()
