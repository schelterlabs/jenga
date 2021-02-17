from sklearn.base import BaseEstimator

from .basis import DataCorruption, Evaluator, ValidationResult


# Evaluate the impact of one or more data corruption on the prediction quality of a model;
# applies the corruptions repeatedly to copies of the test data
# and computes the model's prediction quality for afterwards
class CorruptionImpactEvaluator(Evaluator):

    def evaluate(self, model: BaseEstimator, num_repetitions: int, *corruptions: DataCorruption):

        if not self._task.is_image_data:
            test_data_copy = self._task.test_data.copy(deep=True)
        else:
            test_data_copy = self._task.test_data.copy()

        baseline_predictions = self._model_predict(model, test_data_copy)
        baseline_score = self._task.score_on_test_data(baseline_predictions)

        results = []

        num_results = len(corruptions) * num_repetitions

        current_run = 0
        import time
        t = time.process_time()

        # Evaluate each specified corruption
        for corruption in corruptions:
            corrupted_scores = []
            # Repeatedly
            for _ in range(num_repetitions):

                if not self._task.is_image_data:
                    test_data_copy = self._task.test_data.copy(deep=True)
                else:
                    test_data_copy = self._task.test_data.copy()

                corrupted_data = corruption.transform(test_data_copy)

                # corrupted_predictions = model.predict_proba(corrupted_data)
                corrupted_predictions = self._model_predict(model, corrupted_data)
                corrupted_score = self._task.score_on_test_data(corrupted_predictions)

                corrupted_scores.append(corrupted_score)

                if current_run % 10 == 0:
                    print(f"{current_run}/{num_results} ({time.process_time() - t})")
                current_run += 1

            results.append(ValidationResult(corruption, baseline_score, corrupted_scores))

        return results
