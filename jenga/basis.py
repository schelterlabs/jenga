import numpy as np

from abc import ABC, abstractmethod
from sklearn.metrics import roc_auc_score


class BinaryClassificationTask(ABC):

    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.__test_labels = test_labels

    @abstractmethod
    def fit_baseline_model(self, train_data, train_labels):
        pass

    def score_on_test_data(self, predicted_label_probabilities):
        return roc_auc_score(self.__test_labels, np.transpose(predicted_label_probabilities)[1])
