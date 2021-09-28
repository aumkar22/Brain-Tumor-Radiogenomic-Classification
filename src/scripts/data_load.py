import tensorflow as tf

from abc import ABC, abstractmethod


class DataLoad(ABC):
    @staticmethod
    def __int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def __bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @abstractmethod
    def data_to_tfrecords(self):
        """
        Function to convert data to tfrecords

        :return:
        """
        pass

    @abstractmethod
    def data_load(self):
        """
        Load data

        :return:
        """
        pass

    @abstractmethod
    def perform_preprocess(self, image_volume: tf.Tensor):
        """
        Function to apply preprocessing

        :return:
        """
        pass
