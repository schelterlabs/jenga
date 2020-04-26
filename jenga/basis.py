import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score


class BinaryClassificationTask(ABC):

    def __init__(self,
                 train_data,
                 train_labels,
                 test_data,
                 test_labels,
                 categorical_columns=[],
                 numerical_columns=[],
                 text_columns=[],
                 is_image_data=False
                 ):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.__test_labels = test_labels
        self.categorical_columns = categorical_columns,
        self.numerical_columns = numerical_columns,
        self.text_columns = text_columns,
        self.is_image_data = is_image_data

    @abstractmethod
    def fit_baseline_model(self, train_data, train_labels):
        pass

    def score_on_test_data(self, predicted_label_probabilities):
        return roc_auc_score(self.__test_labels, np.transpose(predicted_label_probabilities)[1])


class DataCorruption:

    @abstractmethod
    def transform(self, data):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"