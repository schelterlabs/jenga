class ValidationResult:

    def __init__(self, corruption, baseline_score, corrupted_scores):
        self.corruption = corruption
        self.baseline_score = baseline_score
        self.corrupted_scores = corrupted_scores

        self.performance_drops_in_percent = [((baseline_score - corrupted_score) / baseline_score) * 100
                                             for corrupted_score in corrupted_scores]


class CorruptionImpactEvaluator:

    def __init__(self, task):
        self.__task = task

    def evaluate(self, model, num_repetitions, *corruptions):

        baseline_predictions = model.predict_proba(self.__task.test_data)
        baseline_score = self.__task.score_on_test_data(baseline_predictions)

        results = []

        for corruption in corruptions:
            corrupted_scores = []
            for _ in range(0, num_repetitions):
                test_data_copy = self.__task.test_data.copy(deep=True)
                corrupted_data = corruption.transform(test_data_copy)

                corrupted_predictions = model.predict_proba(corrupted_data)
                corrupted_score = self.__task.score_on_test_data(corrupted_predictions)

                corrupted_scores.append(corrupted_score)

            results.append(ValidationResult(corruption, baseline_score, corrupted_scores))

        return results
