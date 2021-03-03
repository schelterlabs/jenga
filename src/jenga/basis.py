import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import BINARY_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, REGRESSION


class Task(ABC):
    def __init__(
        self,
        train_data: pd.DataFrame,
        train_labels: pd.Series,
        test_data: pd.DataFrame,
        test_labels: pd.Series,
        categorical_columns: List[str] = [],
        numerical_columns: List[str] = [],
        text_columns: List[str] = [],
        is_image_data: bool = False,
        seed: Optional[int] = None
    ):
        """
        Abstract base class for all Tasks. It defines the interface and a fair amount of functionality, \
            such as fitting a baseline model, inherited to child classes.

        Args:
            train_data (pd.DataFrame): Training data
            train_labels (pd.Series): Training labels
            test_data (pd.DataFrame): Test data
            test_labels (pd.Series): Test labels
            categorical_columns (List[str], optional): List of categorical column names. Defaults to [].
            numerical_columns (List[str], optional): List of numerical column names. Defaults to [].
            text_columns (List[str], optional): List of text column names. Defaults to [].
            is_image_data (bool, optional): Indicates whether data are images. Defaults to False.
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """

        self._baseline_model = None
        self._task_type: Optional[int] = None

        self.train_data = train_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.is_image_data = is_image_data
        self._seed = seed

        # Fix random seeds for reproducibility
        if self._seed:
            random.seed(self._seed)
            np.random.seed(self._seed)

            try:
                import tensorflow as tf
                tf.random.set_seed(self._seed)
            except ImportError:
                pass

    def _get_task_type_of_data(self) -> Optional[int]:
        """
        Helper method to check the given label's task type (classification/multi-class/regression).

        Returns:
            Optional[int]: Integer encoded task type
        """

        task_type = None

        if pd.api.types.is_numeric_dtype(self.train_labels) and pd.api.types.is_numeric_dtype(self.test_labels):
            task_type = REGRESSION

        elif pd.api.types.is_categorical_dtype(self.train_labels) and pd.api.types.is_categorical_dtype(self.test_labels):
            num_test_classes = len(self.test_labels.dtype.categories)
            num_train_classes = len(self.train_labels.dtype.categories)

            if num_test_classes == num_train_classes:

                if num_test_classes == 2:
                    task_type = BINARY_CLASSIFICATION

                elif num_test_classes > 2:
                    task_type = MULTI_CLASS_CLASSIFICATION

        return task_type

    def fit_baseline_model(self, train_data: Optional[pd.DataFrame] = None, train_labels: Optional[pd.Series] = None) -> BaseEstimator:
        """
        Fit a baseline model. If no data is given (default), it uses the task's train data and creates the attribute `_baseline_model`. \
            If data is given, it trains this data.

        Args:
            train_data (Optional[pd.DataFrame], optional): Data to train. Defaults to None.
            train_labels (Optional[pd.Series], optional): Labels to train. Defaults to None.

        Raises:
            ValueError: If `train_data` is given but `train_labels` not or vice versa

        Returns:
            BaseEstimator: Trained model
        """

        if (train_data is None and train_labels is not None) or (train_data is not None and train_labels is None):
            raise ValueError("either set both parameters (train_data, train_labels) or non")

        use_original_data = train_data is None

        # shortcut if model is already trained
        if use_original_data and self._baseline_model:
            return self._baseline_model

        if use_original_data:
            train_data = self.train_data.copy()
            train_labels = self.train_labels.copy()

        for col in self.categorical_columns:
            train_data[col] = train_data[col].astype(str)

        categorical_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='most_frequent')),
                ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        numerical_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='mean')),
                ('scaling',  StandardScaler())
            ]
        )

        feature_transformation = ColumnTransformer(transformers=[
                ('categorical_features', categorical_preprocessing, self.categorical_columns),
                ('scaled_numeric', numerical_preprocessing, self.numerical_columns)
            ]
        )

        param_grid, pipeline, scorer = self._get_pipeline_grid_scorer_tuple(feature_transformation)
        refit = list(scorer.keys())[0]

        search = GridSearchCV(pipeline, param_grid, scoring=scorer, n_jobs=-1, refit=refit)
        model = search.fit(train_data, train_labels).best_estimator_

        # only set baseline model attribute if it is trained on the original task data
        if use_original_data:
            self._baseline_model = model

        return model

    @abstractmethod
    def _check_data(self):
        """
        Forces child class to implement data checking.

        Raises:
            Exception: If child class does not set the attribute `_task_type`
        """

        # TODO: maybe we also want to check whether all the column names exist in the dataframe

        if self._task_type is None:
            raise Exception("Class attribute '_task_type' is not set!")

    @abstractmethod
    def get_baseline_performance(self) -> float:
        """
        Forces child class to implement baseline performance metric.

        Raises:
            Exception: If no baseline model is trained yet

        Returns:
            float: Baseline performance on test data
        """

        if not self._baseline_model:
            raise Exception("First fit a baseline model")

    @abstractmethod
    def score_on_test_data(self, predictions: pd.array) -> float:
        """
        Forces child class to implement scoring of given `predictions` against test data.

        Args:
            predictions (pd.array): Either 1-D array if models `predict` method used or n-D array for `predict_proba`, \
                where n is the number of classes

        Returns:
            float: Score of given `predictions`
        """

        pass

    @abstractmethod
    def _get_pipeline_grid_scorer_tuple(self, feature_transformation: ColumnTransformer) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:
        """
        Forces child class to define task specific `Pipeline`, hyperparameter grid for HPO, and scorer for baseline model training.
        This helps to reduce redundant code.

        Args:
            feature_transformation (ColumnTransformer): Basic preprocessing for columns. Given by `fit_baseline_model` that calls this method

        Returns:
            Tuple[Dict[str, object], Any, Dict[str, Any]]: Task specific parts to build baseline model
        """

        pass


class BinaryClassificationTask(Task):

    def __init__(
        self,
        train_data: pd.DataFrame,
        train_labels: pd.Series,
        test_data: pd.DataFrame,
        test_labels: pd.Series,
        categorical_columns: List[str] = [],
        numerical_columns: List[str] = [],
        text_columns: List[str] = [],
        is_image_data: bool = False,
        seed: Optional[int] = None
    ):
        """
        Class that represents a binary classification task. \
            If `is_image_data = False` it forces the `train_labels` and `test_labels` to be of a `categorical_dtype`.
        It implements abstract methods defined by parent class `Task`.

        Args:
            train_data (pd.DataFrame): Training data
            train_labels (pd.Series): Training labels
            test_data (pd.DataFrame): Test data
            test_labels (pd.Series): Test labels
            categorical_columns (List[str], optional): List of categorical column names. Defaults to [].
            numerical_columns (List[str], optional): List of numerical column names. Defaults to [].
            text_columns (List[str], optional): List of text column names. Defaults to [].
            is_image_data (bool, optional): Indicates whether data are images. Defaults to False.
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            text_columns=text_columns,
            is_image_data=is_image_data,
            seed=seed
        )

        self._task_type = BINARY_CLASSIFICATION
        self._check_data()

    def _check_data(self):
        """
        Checks whether or not the given data/labels are of `categorical_dtype`.

        Raises:
            ValueError: If labels does not fit the constrains for a binary classification task.
        """

        super()._check_data()

        if self._get_task_type_of_data() != BINARY_CLASSIFICATION and not self.is_image_data:
            raise ValueError("Downloaded data is not a binary classification task.")

    def _get_pipeline_grid_scorer_tuple(
        self,
        feature_transformation: ColumnTransformer
    ) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:
        """
        Binary classification specific default `Pipeline`, hyperparameter grid for HPO, and scorer for baseline model training.

        Args:
            feature_transformation (ColumnTransformer): Basic preprocessing for columns. Given by `fit_baseline_model` that calls this method

        Returns:
            Tuple[Dict[str, object], Any, Dict[str, Any]]: Binary classification specific parts to build baseline model
        """

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2'],
            'learner__alpha': [0.00001, 0.0001, 0.001, 0.01]
        }

        pipeline = Pipeline(
            [
                ('features', feature_transformation),
                ('learner', SGDClassifier(max_iter=1000, n_jobs=-1))
            ]
        )

        scorer = {
            "ROC/AUC": make_scorer(roc_auc_score, needs_proba=True)
        }

        return param_grid, pipeline, scorer

    def get_baseline_performance(self) -> float:
        """
        By default calculate the ROC/AUC score of the baseline model based on test data.

        Returns:
            float: Baseline performance on test data
        """

        super().get_baseline_performance()

        predicted_label_probabilities = self._baseline_model.predict_proba(self.test_data)
        return self.score_on_test_data(predicted_label_probabilities)

    def score_on_test_data(self, predictions: pd.array) -> float:
        """
        By default calculate the ROC/AUC score of the given `predictions` against test data.

        Args:
            predictions (pd.array): n-D array given by a model's `predict_proba` method, where n is the number of classes

        Returns:
            float: ROC/AUC score of given `predictions`
        """

        return roc_auc_score(self.test_labels, predictions[:, 1])


class MultiClassClassificationTask(Task):

    def __init__(
        self,
        train_data: pd.DataFrame,
        train_labels: pd.Series,
        test_data: pd.DataFrame,
        test_labels: pd.Series,
        categorical_columns: List[str] = [],
        numerical_columns: List[str] = [],
        text_columns: List[str] = [],
        is_image_data: bool = False,
        seed: Optional[int] = None
    ):
        """
        Class that represents a multi-class classification task. \
            If `is_image_data = False` it forces the `train_labels` and `test_labels` to be of a `categorical_dtype`.
        It implements abstract methods defined by parent class `Task`.

        Args:
            train_data (pd.DataFrame): Training data
            train_labels (pd.Series): Training labels
            test_data (pd.DataFrame): Test data
            test_labels (pd.Series): Test labels
            categorical_columns (List[str], optional): List of categorical column names. Defaults to [].
            numerical_columns (List[str], optional): List of numerical column names. Defaults to [].
            text_columns (List[str], optional): List of text column names. Defaults to [].
            is_image_data (bool, optional): Indicates whether data are images. Defaults to False.
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            text_columns=text_columns,
            is_image_data=is_image_data,
            seed=seed
        )

        self._task_type = MULTI_CLASS_CLASSIFICATION
        self._check_data()

    def _check_data(self):
        """
        Checks whether or not the given data/labels are of `categorical_dtype`.

        Raises:
            ValueError: If labels does not fit the constrains for a multi-class classification task.
        """

        super()._check_data()

        if self._get_task_type_of_data() != MULTI_CLASS_CLASSIFICATION and not self.is_image_data:
            raise ValueError("Downloaded data is not a multi-class classification task.")

    def _get_pipeline_grid_scorer_tuple(
        self,
        feature_transformation: ColumnTransformer
    ) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:
        """
        Multi-class classification specific default `Pipeline`, hyperparameter grid for HPO, and scorer for baseline model training.

        Args:
            feature_transformation (ColumnTransformer): Basic preprocessing for columns. Given by `fit_baseline_model` that calls this method

        Returns:
            Tuple[Dict[str, object], Any, Dict[str, Any]]: Multi-class classification specific parts to build baseline model
        """

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2'],
            'learner__alpha': [0.00001, 0.0001, 0.001, 0.01]
        }

        pipeline = Pipeline(
            [
                ('features', feature_transformation),
                ('learner', SGDClassifier(max_iter=1000, n_jobs=-1))
            ]
        )

        scorer = {
            "F1": make_scorer(f1_score, average="macro")
        }

        return param_grid, pipeline, scorer

    def get_baseline_performance(self) -> float:
        """
        By default calculate the F1 score of the baseline model based on test data.

        Calculate the ROC/AUC value for a test set that does not contains samples from all possible classes is not supported by `roc_auc_score`. \
            This is why default score differs from the binary-classification score.

        Returns:
            float: Baseline performance on test data
        """

        super().get_baseline_performance()

        predictions = self._baseline_model.predict(self.test_data)
        return self.score_on_test_data(predictions)

    def score_on_test_data(self, predictions: pd.array) -> float:
        """
        By default calculate the F1 score of the given `predictions` against test data.

        Calculate the ROC/AUC value for a test set that does not contains samples from all possible classes is not supported by `roc_auc_score`. \
            This is why the default score differs from the binary-classification score.

        Args:
            predictions (pd.array): 1-D array given by a model's `predict` method

        Returns:
            float: F1 score of given `predictions`
        """

        return f1_score(self.test_labels, predictions, average="macro")


class RegressionTask(Task):

    def __init__(
        self,
        train_data: pd.DataFrame,
        train_labels: pd.Series,
        test_data: pd.DataFrame,
        test_labels: pd.Series,
        categorical_columns: List[str] = [],
        numerical_columns: List[str] = [],
        text_columns: List[str] = [],
        is_image_data: bool = False,
        seed: Optional[int] = None
    ):
        """
        Class that represents a regression task. Forces the `train_labels` and `test_labels` to be of a `numeric_dtype`.
        It implements abstract methods defined by parent class `Task`.

        Args:
            train_data (pd.DataFrame): Training data
            train_labels (pd.Series): Training labels
            test_data (pd.DataFrame): Test data
            test_labels (pd.Series): Test labels
            categorical_columns (List[str], optional): List of categorical column names. Defaults to [].
            numerical_columns (List[str], optional): List of numerical column names. Defaults to [].
            text_columns (List[str], optional): List of text column names. Defaults to [].
            is_image_data (bool, optional): Indicates whether data are images. Defaults to False.
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            text_columns=text_columns,
            is_image_data=is_image_data,
            seed=seed
        )

        self._task_type = REGRESSION
        self._check_data()

    def _check_data(self):
        """
        Checks whether or not the given data/labels are of `numeric_dtype`.

        Raises:
            ValueError: If labels does not fit the constrains for a regression task.
        """

        super()._check_data()

        if self._get_task_type_of_data() != REGRESSION:
            raise ValueError("Downloaded data is not a regression task.")

    def _get_pipeline_grid_scorer_tuple(
        self,
        feature_transformation: ColumnTransformer
    ) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:
        """
        Regression specific default `Pipeline`, hyperparameter grid for HPO, and scorer for baseline model training.

        Args:
            feature_transformation (ColumnTransformer): Basic preprocessing for columns. Given by `fit_baseline_model` that calls this method

        Returns:
            Tuple[Dict[str, object], Any, Dict[str, Any]]: Regression specific parts to build baseline model
        """

        param_grid = {
            'learner__loss': ['squared_loss', 'huber'],
            'learner__penalty': ['l2'],
            'learner__alpha': [0.00001, 0.0001, 0.001, 0.01]
        }

        pipeline = Pipeline(
            [
                ('features', feature_transformation),
                ('learner', SGDRegressor(max_iter=1000))
            ]
        )

        scorer = {
            "MSE": make_scorer(mean_squared_error, greater_is_better=False),
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False)
        }

        return param_grid, pipeline, scorer

    def get_baseline_performance(self) -> float:
        """
        By default calculate the MSE of the baseline model based on test data.

        Returns:
            float: Baseline performance on test data
        """

        super().get_baseline_performance()

        predictions = self._baseline_model.predict(self.test_data)
        return self.score_on_test_data(predictions)

    def score_on_test_data(self, predictions: pd.array) -> float:
        """
        By default calculate the MSE of the given `predictions` against test data.

        Args:
            predictions (pd.array): 1-D array given by a model's `predict` method

        Returns:
            float: MSE of given `predictions`
        """

        return mean_squared_error(self.test_labels, predictions)


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
