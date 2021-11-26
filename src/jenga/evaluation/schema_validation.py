import tensorflow_data_validation as tfdv
from sklearn.base import BaseEstimator

from ..basis import DataCorruption
from .basis import Evaluator, ValidationResult


class SchemaValidationResult(ValidationResult):

    def __init__(self, corruption, anomalies, baseline_score, corrupted_scores):

        super().__init__(
            corruption=corruption,
            baseline_score=baseline_score,
            corrupted_scores=corrupted_scores
        )
        self.anomalies_detected = [len(exp_anomalies) > 0 for exp_anomalies in anomalies]


# Takes a tfdv schema and corruptions to evaluate two things at once:
# (1) Does tfdv detect an anomaly for the corruption?
# (2) How much does the corruption decrease the prediction quality
class SchemaValidationEvaluator(Evaluator):

    # Auto-infer a schema from the training daya
    def schema_from_train_data(self):
        train_data_stats = tfdv.generate_statistics_from_dataframe(self._task.train_data)
        schema = tfdv.infer_schema(statistics=train_data_stats)
        return schema

    def evaluate(self, model: BaseEstimator, num_repetitions: int, *corruptions: DataCorruption):

        schema = self.schema_from_train_data()

        baseline_predictions = model.predict_proba(self._task.test_data)
        baseline_score = self._task.score_on_test_data(baseline_predictions)

        results = []

        # Repeatedly corrupt the test data
        for corruption in corruptions:
            corrupted_scores = []
            anomalies = []
            for _ in range(0, num_repetitions):
                test_data_copy = self._task.test_data.copy(deep=True)
                corrupted_data = corruption.transform(test_data_copy)

                # Determine whether tfdv finds anomalies in the data
                corrupted_data_stats = tfdv.generate_statistics_from_dataframe(corrupted_data)
                tfdv_anomalies = tfdv.validate_statistics(statistics=corrupted_data_stats, schema=schema)

                schema_anomalies = tfdv_anomalies.anomaly_info

                # Compute the prediction score on the test data
                corrupted_predictions = model.predict_proba(corrupted_data)
                corrupted_score = self._task.score_on_test_data(corrupted_predictions)

                anomalies.append(schema_anomalies)
                corrupted_scores.append(corrupted_score)

            results.append(SchemaValidationResult(corruption, anomalies, baseline_score, corrupted_scores))

        return results
