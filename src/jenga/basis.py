import random
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score


# Base class for binary classification tasks, including training data, test data, a baseline model and scoring
class BinaryClassificationTask(ABC):

    def __init__(
        self,
        seed,
        train_data,
        train_labels,
        test_data,
        test_labels,
        categorical_columns=None,
        numerical_columns=None,
        text_columns=None,
        is_image_data=False
    ):

        if numerical_columns is None:
            numerical_columns = []
        if categorical_columns is None:
            categorical_columns = []
        if text_columns is None:
            text_columns = []

        # Fix random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Train and test data and labels
        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.__test_labels = test_labels

        # Information about the data (column types, etc)
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.is_image_data = is_image_data

    # Abstract base method for training a baseline model on the raw data
    @abstractmethod
    def fit_baseline_model(self, train_data, train_labels):
        pass

    # Per default, we compute ROC AUC scores
    def score_on_test_data(self, predicted_label_probabilities):
        return roc_auc_score(self.__test_labels, np.transpose(predicted_label_probabilities)[1])


# Abstract base class for all data corruptions
class DataCorruption:

    # Abstract base method for corruptions, they have to return a corrupted copied of the dataframe
    @abstractmethod
    def transform(self, data):
        pass

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"


class TabularCorruption(DataCorruption):
    def __init__(self, column, fraction, sampling='CAR'):
        '''
        Corruptions for structured data
        Input:
        column:    column to perturb, string
        fraction:   fraction of rows to corrupt, float between 0 and 1
        sampling:   sampling mechanism for corruptions, options are completely at random ('CAR'),
                     at random ('AR'), not at random ('NAR')
        '''
        self.column = column
        self.fraction = fraction
        self.sampling = sampling

    def get_dtype(self, df):
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
        return numeric_cols, non_numeric_cols

    def sample_rows(self, data):

        # Completely At Random
        if self.sampling.endswith('CAR'):
            rows = np.random.permutation(data.index)[:int(len(data)*self.fraction)]
        elif self.sampling.endswith('NAR') or self.sampling.endswith('AR'):
            n_values_to_discard = int(len(data) * min(self.fraction, 1.0))
            perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
            perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

            # Not At Random
            if self.sampling.endswith('NAR'):
                # pick a random percentile of values in this column
                rows = data[self.column].sort_values().iloc[perc_idx].index

            # At Random
            elif self.sampling.endswith('AR'):
                depends_on_col = np.random.choice(list(set(data.columns) - {self.column}))
                # pick a random percentile of values in other column
                rows = data[depends_on_col].sort_values().iloc[perc_idx].index

        else:
            ValueError(f"sampling type '{self.sampling}' not recognized")

        return rows
