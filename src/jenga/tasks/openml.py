from typing import Any, Dict, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..basis import (
    BinaryClassificationTask,
    MultiClassClassificationTask,
    RegressionTask,
    Task
)


class OpenMLTask(Task):

    def __init__(self, openml_id: int, train_size: float = 0.8, seed: Optional[int] = None):
        """
        Base class for task that get data from [OpenML](https://www.openml.org).

        Args:
            openml_id (int): ID of the to-be-fetched data from [OpenML](https://www.openml.org)
            train_size (float, optional): Defines the data split. Defaults to 0.8.
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """

        X, y = fetch_openml(data_id=openml_id, as_frame=True, return_X_y=True)
        train_data, test_data, train_labels, test_labels = train_test_split(X, y, train_size=train_size)

        categorical_columns = [
            column for column in X.columns
            if pd.api.types.is_categorical_dtype(X[column])
        ]
        numerical_columns = [
            column for column in X.columns
            if pd.api.types.is_numeric_dtype(X[column]) and column not in categorical_columns
        ]

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            is_image_data=False,
            seed=seed
        )


class OpenMLRegressionTask(OpenMLTask, RegressionTask):
    """
    Class that represents a regression task and gets data from [OpenML](https://www.openml.org).
    """

    def _get_pipeline_grid_scorer_tuple(
        self,
        feature_transformation: ColumnTransformer
    ) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:

        param_grid = {
            'learner__loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01]
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


class OpenMLMultiClassClassificationTask(OpenMLTask, MultiClassClassificationTask):
    """
    Class that represents a multi-class classification task and gets data from [OpenML](https://www.openml.org).
    """

    def _get_pipeline_grid_scorer_tuple(
        self,
        feature_transformation: ColumnTransformer
    ) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:

        param_grid = {
            'learner__loss': ['log', 'modified_huber'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01]
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


class OpenMLBinaryClassificationTask(OpenMLTask, BinaryClassificationTask):
    """
    Class that represents a binary classification task and gets data from [OpenML](https://www.openml.org).
    """

    def _get_pipeline_grid_scorer_tuple(
        self,
        feature_transformation: ColumnTransformer
    ) -> Tuple[Dict[str, object], Any, Dict[str, Any]]:

        param_grid = {
            'learner__loss': ['log', 'modified_huber'],
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01]
        }

        pipeline = Pipeline(
            [
                ('features', feature_transformation),
                ('learner', SGDClassifier(max_iter=1000, n_jobs=-1))
            ]
        )

        scorer = {
            "F1": make_scorer(f1_score, average="macro"),
            "ROC/AUC": make_scorer(roc_auc_score, needs_proba=True)
        }

        return param_grid, pipeline, scorer
