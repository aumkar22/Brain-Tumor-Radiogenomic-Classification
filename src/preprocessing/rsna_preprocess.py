import tensorflow as tf
import pandas as pd
import numpy as np

from typing import Tuple, List
from sklearn.model_selection import GroupKFold


@tf.function
def normalization(image_volume: tf.Tensor) -> tf.Tensor:

    """
    Image normalization by subtracting mean and dividing by standard deviation

    :param image_volume: Input image volume tensor
    :return: Normalized tensor
    """

    normalized_image = (
        image_volume - tf.reduce_mean(image_volume)
    ) / tf.math.reduce_std(image_volume)

    return normalized_image


@tf.function
def tf_equalize_histogram(image_volume: tf.Tensor) -> tf.Tensor:

    """
    Contrast enhancement using histogram equalization. Code taken from:
    https://stackoverflow.com/questions/42835247/how-to-implement-histogram-equalization-for-images-in-tensorflow

    :param image_volume:
    :return:
    """

    values_range = tf.constant([0.0, 255.0], dtype=tf.float32)
    histogram = tf.histogram_fixed_width(
        tf.compat.v1.to_float(image_volume), values_range, 256
    )
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image_volume)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(
        tf.compat.v1.to_float(cdf - cdf_min)
        * 255.0
        / tf.compat.v1.to_float(pix_cnt - 1)
    )
    px_map = tf.cast(px_map, tf.uint8)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image_volume, tf.int32)), 2)
    return eq_hist


def data_split(
    data_df: pd.DataFrame, group: str, n_splits: int = 5
) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    """
    Function to split the data into training and validation sets.
    Seed not added as GroupKFold is not randomized.

    :param data_df: Input dataframe
    :param group: Groups on which split should be made
    :param n_splits: Number of folds for cross validation
    :return:
    """

    group_fold = GroupKFold(n_splits=n_splits)
    train_indices = []
    validation_indices = []

    for train_index, val_index in enumerate(
        group_fold.split(data_df, groups=data_df[group])
    ):
        train_indices.append(train_index)
        validation_indices.append(val_index)

    return train_indices, validation_indices
