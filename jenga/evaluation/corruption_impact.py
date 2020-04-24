import tensorflow_data_validation as tfdv


class ValidationResult:

    def __init__(self, corruption, baseline_score, corrupted_score):
        self.corruption = corruption
        self.baseline_score = baseline_score
        self.corrupted_score = corrupted_score

        self.performance_drop_in_percent = ((baseline_score - corrupted_score) / baseline_score) * 100


class CorruptionImpactEvaluator:

    def __init__(self, task):
        self.__task = task

    def evaluate(self, model, *corruptions):

        baseline_predictions = model.predict_proba(self.__task.test_data)
        baseline_score = self.__task.score_on_test_data(baseline_predictions)

        results = []

        for corruption in corruptions:
            test_data_copy = self.__task.test_data.copy(deep=True)
            corrupted_data = corruption.transform(test_data_copy)

            corrupted_predictions = model.predict_proba(corrupted_data)
            corrupted_score = self.__task.score_on_test_data(corrupted_predictions)

            results.append(ValidationResult(corruption, baseline_score, corrupted_score))

        return results
