class AutoCleaningValidationResult:

    def __init__(self, 
                 corruption, 
                 cleaner, 
                 baseline_score, 
                 corrupted_scores, 
                 score_with_anomaly_removal,
                 score_with_imputation,
                 score_with_cleaning):
        self.corruption = corruption
        self.cleaner = cleaner
        self.baseline_score = baseline_score
        self.corrupted_scores = corrupted_scores
        self.score_with_anomaly_removal = score_with_anomaly_removal
        self.score_with_imputation = score_with_imputation
        self.score_with_cleaning = score_with_cleaning

# Evaluate the impact of one or more data corruption on the prediction quality of a model;
# applies the corruptions repeatedly to copies of the test data
# and computes the model's prediction quality for afterwards
class AutoCleaningEvaluator:

    def __init__(self, task, cleaner):
        self.__task = task
        self.__cleaner = cleaner

    def compute_eval_score(self, model, test_data):
        predictions = model.predict_proba(test_data)
        print(self.__task.score_on_test_data(predictions))
        return self.__task.score_on_test_data(predictions)

    def evaluate(self, model, num_repetitions, *corruptions):
        
        test_data_copy = self.__task.test_data.copy()

        baseline_score = self.compute_eval_score(model, test_data_copy)

        results = []

        num_results = len(corruptions) * num_repetitions

        current_run = 0
        import time
        t = time.process_time()

        # Evaluate each specified corruption
        for corruption in corruptions:
            corrupted_scores = []
            scores_with_anomaly_removal = []
            scores_with_imputation = []
            scores_with_cleaning = []

            # Repeatedly
            for _ in range(0, num_repetitions):

                test_data_copy = self.__task.test_data.copy(deep=True)
                
                corrupted_data = corruption.transform(test_data_copy)
                corrupted_scores.append(self.compute_eval_score(model, corrupted_data))

                df_outliers_removed = self.__cleaner.remove_outliers(corrupted_data)
                scores_with_anomaly_removal.append(self.compute_eval_score(model, df_outliers_removed))

                df_imputed = self.__cleaner.impute(corrupted_data)
                scores_with_imputation.append(self.compute_eval_score(model, df_imputed))

                df_cleaned = self.__cleaner.impute(df_outliers_removed)
                scores_with_cleaning.append(self.compute_eval_score(model, df_cleaned))

                if current_run % 10 == 0:
                    print(f"{current_run}/{num_results} ({time.process_time() - t})")
                current_run += 1

            results.append(
                AutoCleaningValidationResult(corruption, 
                                 self.__cleaner,
                                 baseline_score, 
                                 corrupted_scores,
                                 scores_with_anomaly_removal,
                                 scores_with_imputation, 
                                 scores_with_cleaning))

        return results

