from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..basis import DataCorruption, Task
from ..utils import BINARY_CLASSIFICATION, MULTI_CLASS_CLASSIFICATION, REGRESSION


class ValidationResult:

    def __init__(self, corruption, baseline_score, corrupted_scores):
        self.corruption = corruption
        self.baseline_score = baseline_score
        self.corrupted_scores = corrupted_scores


class Evaluator(ABC):
    def __init__(self, task: Task):
        self._task = task

    def _model_predict(self, model: BaseEstimator, data: pd.DataFrame) -> np.array:

        if self._task._task_type == BINARY_CLASSIFICATION:
            predictions = model.predict_proba(data)

        elif self._task._task_type == MULTI_CLASS_CLASSIFICATION:
            predictions = model.predict(data)

        elif self._task._task_type == REGRESSION:
            predictions = model.predict(data)

        return predictions

    @abstractmethod
    def evaluate(self, model: BaseEstimator, num_repetitions: int, *corruptions: DataCorruption):
        pass
