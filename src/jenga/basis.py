import random
from typing import Optional
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from typing import Tuple, Dict, List, Union, Callable, Any


class OpenMLTask(ABC):
    def __init__(self, openml_id: int, seed: Optional[int] = None):

        self._seed = seed
        self._openml_id = openml_id
        self._baseline_model = None

        # Fix random seeds for reproducibility
        if self._seed:
            random.seed(self._seed)
            np.random.seed(self._seed)
            tf.random.set_seed(self._seed)

        X, y = fetch_openml(data_id=self._openml_id, as_frame=True, return_X_y=True)

        self.__contains_missing_values = X.isnull().values.any()  # TODO: does this work in all cases?

        self.categorical_columns, self.numerical_columns, self.text_columns = self._guess_dtypes(X)
        print(f'Found {len(self.categorical_columns)} categorical columns: {self.categorical_columns}')
        print(f'Found {len(self.numerical_columns)} numeric columns: {self.numerical_columns}')

        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(X, y, test_size=0.2)

        self._check_data()

    def _is_categorical(self, col, max_unique_ratio=0.05):
        # return len(col.value_counts()) / len(col) < max_unique_ratio
        return pd.api.types.is_categorical_dtype(col)

    def _guess_dtypes(self, df):
        categorical_columns = [c for c in df.columns if self._is_categorical(df[c])]
        numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in categorical_columns]
        text_columns = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) and c not in categorical_columns and c not in numeric_columns]

        return categorical_columns, numeric_columns, text_columns

    def _get_task_type(self) -> Optional[str]:

        task_type = None

        if pd.api.types.is_numeric_dtype(self.train_labels):
            task_type = "regression"

        elif pd.api.types.is_categorical_dtype(self.train_labels):
            num_classes = len(self.train_labels.dtype.categories)

            if num_classes == 2:
                task_type = "binary classification"

            elif num_classes > 2:
                task_type = "multilabel classification"

        return task_type

    def contains_missing_values(self) -> bool:
        return self.__contains_missing_values

    def fit_baseline_model(self):
        train_data = self.train_data.copy()
        train_labels = self.train_labels.copy()

        for col in self.categorical_columns:
            train_data[col] = train_data[col].astype(str)

        categorical_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='constant', fill_value='__NA__')),
                ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        numerical_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaling',  StandardScaler())
            ]
        )

        feature_transformation = ColumnTransformer(transformers=[
                ('categorical_features', categorical_preprocessing, self.categorical_columns),
                ('scaled_numeric', numerical_preprocessing, self.numerical_columns)
            ]
        )

        param_grid, pipeline, scorer = self._get_pipeline_grid_scorer(feature_transformation)
        refit = list(scorer.keys())[0]

        search = GridSearchCV(pipeline, param_grid, scoring=scorer, n_jobs=-1, refit=refit)
        self._baseline_model = search.fit(train_data, train_labels).best_estimator_

        return self._baseline_model

    # Check whether data fits the expected form, if not raise error
    @abstractmethod
    def _check_data(self):
        pass

    # Abstract base method for calculating scores
    @abstractmethod
    def score_on_test_data(self):

        if not self._baseline_model:
            raise Exception("First fit a baseline model")

    @abstractmethod
    def _get_pipeline_grid_scorer(
        self,
        feature_transformation: ColumnTransformer
    ) -> Tuple[Dict[str, List[Union[str, float]]], Pipeline, Dict[str, Callable[..., Any]]]:
        pass


class ClassificationTask(OpenMLTask):

    def __init__(
        self,
        openml_id: int,
        is_image_data: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__(openml_id=openml_id, seed=seed)

        self.is_image_data = is_image_data


class BinaryClassificationTask(ClassificationTask):

    def _check_data(self):
        if self._get_task_type() != "binary classification":
            raise ValueError("Downloaded data is not a binary classification task.")

    def score_on_test_data(self):

        super().score_on_test_data()

        f1_test_score = roc_test_score = None

        if hasattr(self._baseline_model, "predict_proba"):
            predicted_label_probabilities = self._baseline_model.predict_proba(self.test_data)
            roc_test_score = roc_auc_score(self.test_labels, predicted_label_probabilities[:, 1])

        predictions = self._baseline_model.predict(self.test_data)
        f1_test_score = f1_score(self.test_labels, predictions, average="macro")

        return {"ROC/AUC": roc_test_score, "F1": f1_test_score}


class MultiLabelClassificationTask(ClassificationTask):

    def _check_data(self):
        if self._get_task_type() != "multilabel classification":
            raise ValueError("Downloaded data is not a multi-label classification task.")

    def score_on_test_data(self):

        super().score_on_test_data()

        f1_test_score = roc_test_score = None

        if hasattr(self._baseline_model, "predict_proba"):
            # NOTE: ROC/AUC score is problematic for many classes with just a few samples per class
            # it only works if for each class at least one examples exists. Due to sampling the probability fo causing an error
            # is very high, so catching this
            try:
                predicted_label_probabilities = self._baseline_model.predict_proba(self.test_data)
                roc_test_score = roc_auc_score(self.test_labels, predicted_label_probabilities, multi_class="ovo")

            except ValueError:
                pass

        predictions = self._baseline_model.predict(self.test_data)
        f1_test_score = f1_score(self.test_labels, predictions, average="macro")

        return {"ROC/AUC": roc_test_score, "F1": f1_test_score}


class RegressionTask(OpenMLTask):

    def _check_data(self):
        if self._get_task_type() != "regression":
            raise ValueError("Downloaded data is not a regression task.")

    def score_on_test_data(self):

        if not self._baseline_model:
            raise Exception("First fit a baseline model")

        predictions = self._baseline_model.predict(self.test_data)

        return {
            "MSE": mean_squared_error(self.test_labels, predictions),
            "MAE": mean_absolute_error(self.test_labels, predictions)
        }


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
