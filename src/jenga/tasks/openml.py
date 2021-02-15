import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, mean_squared_error, mean_absolute_error

from typing import Tuple, Dict, List, Union, Callable, Any

from jenga.basis import BinaryClassificationTask, MultiLabelClassificationTask, RegressionTask


class OpenMLRegressionTask(RegressionTask):

    def _get_pipeline_grid_scorer(self, feature_transformation: ColumnTransformer) -> Tuple[Dict[str, List[Union[str, float]]], Pipeline, Dict[str, Callable[..., Any]]]:

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


class OpenMLMultiLabelClassificationTask(MultiLabelClassificationTask):

    def _get_pipeline_grid_scorer(self, feature_transformation: ColumnTransformer) -> Tuple[Dict[str, List[Union[str, float]]], Pipeline, Dict[str, Callable[..., Any]]]:

        param_grid = {
            'learner__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
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


class OpenMLBinaryClassificationTask(BinaryClassificationTask):

    def _get_pipeline_grid_scorer(self, feature_transformation: ColumnTransformer) -> Tuple[Dict[str, List[Union[str, float]]], Pipeline, Dict[str, Callable[..., Any]]]:

        param_grid = {
            'learner__loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
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
            "ROC/AUC": make_scorer(roc_auc_score, needs_proba=True)
        }

        return param_grid, pipeline, scorer
