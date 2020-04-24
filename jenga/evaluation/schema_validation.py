import tensorflow_data_validation as tfdv


class SchemaValidationResult:

    def __init__(self, corruption, anomalies, baseline_score, corrupted_score):
        self.corruption = corruption
        self.anomalies = anomalies
        self.anomalies_detected = len(anomalies) > 0
        self.baseline_score = baseline_score
        self.corrupted_score = corrupted_score

        self.performance_drop_in_percent = ((baseline_score - corrupted_score) / baseline_score) * 100


class SchemaValidationEvaluator:

    def __init__(self, task):
        self.__task = task

    def schema_from_train_data(self):
        train_data_stats = tfdv.generate_statistics_from_dataframe(self.__task.train_data)
        schema = tfdv.infer_schema(statistics=train_data_stats)
        return schema

    def evaluate_validation(self, model, schema, *corruptions):

        baseline_predictions = model.predict_proba(self.__task.test_data)
        baseline_score = self.__task.score_on_test_data(baseline_predictions)

        results = []

        for corruption in corruptions:
            test_data_copy = self.__task.test_data.copy(deep=True)
            corrupted_data = corruption.transform(test_data_copy)

            corrupted_data_stats = tfdv.generate_statistics_from_dataframe(corrupted_data)
            anomalies = tfdv.validate_statistics(statistics=corrupted_data_stats, schema=schema)

            schema_anomalies = anomalies.anomaly_info

            corrupted_predictions = model.predict_proba(corrupted_data)
            corrupted_score = self.__task.score_on_test_data(corrupted_predictions)

            results.append(SchemaValidationResult(corruption, schema_anomalies, baseline_score, corrupted_score))

        return results
