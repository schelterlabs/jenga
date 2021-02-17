import numpy as np
import pandas as pd
import tensorflow_data_validation as tfdv

from ..corruptions.generic import MissingValues, SwappedValues
from ..corruptions.numerical import GaussianNoise, Scaling
from ..corruptions.text import BrokenCharacters


# Takes a tfdv schema and corruptions to evaluate two things at once:
# (1) Does tfdv detect an anomaly for the corruption?
# (2) How much does the corruption decrease the prediction quality
# This class randomly generates data corruptions though for a given task
class SchemaStresstest:

    def validate(self, schema, data):
        stats = tfdv.generate_statistics_from_dataframe(data)
        return tfdv.validate_statistics(statistics=stats, schema=schema)

    def has_anomaly(self, validation_result):
        return len(validation_result.anomaly_info) != 0

    def run(self, task, model, schema, num_corruptions, performance_threshold):
        # Make sure the schema works on the clean test data
        assert(not self.has_anomaly(self.validate(schema, task.test_data)))

        baseline_predictions = model.predict_proba(task.test_data)
        baseline_score = task.score_on_test_data(baseline_predictions)

        random_corruptions = set()

        for _ in range(0, num_corruptions):
            num_columns = len(task.numerical_columns + task.categorical_columns + task.text_columns)
            p_numerical_column_affected = float(len(task.numerical_columns)) / num_columns
            p_categorical_column_affected = float(len(task.categorical_columns)) / num_columns
            p_text_column_affected = float(len(task.text_columns)) / num_columns

            affected_column_type = np.random.choice(
                ['numerical', 'categorical', 'text'],
                1,
                p=[p_numerical_column_affected, p_categorical_column_affected, p_text_column_affected]
            )

            fraction = float(np.random.randint(100)) / 100

            if affected_column_type == 'numerical':

                if len(task.numerical_columns) >= 2 and np.random.uniform() < 0.1:
                    affected_columns = np.random.choice(task.numerical_columns, 2)
                    random_corruptions.add(SwappedValues(affected_columns[0], swap_with=affected_columns[1], fraction=fraction))
                else:

                    corruption_type = np.random.choice(['missing', 'noise', 'scaling'])

                    if corruption_type == 'missing':
                        missingness = np.random.choice(['MCAR', 'MAR', 'MNAR'])
                        affected_column = np.random.choice(task.numerical_columns)
                        random_corruptions.add(MissingValues(affected_column, fraction, na_value=np.nan, missingness=missingness))

                    elif corruption_type == 'noise':
                        affected_column = np.random.choice(task.numerical_columns)
                        random_corruptions.add(GaussianNoise(affected_column, fraction))

                    elif corruption_type == 'scaling':
                        affected_column = np.random.choice(task.numerical_columns)
                        random_corruptions.add(Scaling(affected_column, fraction))

            elif affected_column_type == 'categorical':

                if len(task.categorical_columns) >= 2 and np.random.uniform() < 0.1:
                    affected_columns = np.random.choice(task.categorical_columns, 2)
                    random_corruptions.add(SwappedValues(affected_columns[0], swap_with=affected_columns[1], fraction=fraction))

                else:
                    corruption_type = np.random.choice(['missing', 'encoding'])

                    if corruption_type == 'missing':
                        missingness = np.random.choice(['MCAR', 'MAR', 'MNAR'])
                        affected_column = np.random.choice(task.categorical_columns)
                        random_corruptions.add(MissingValues(affected_column, fraction, na_value='', missingness=missingness))

                    elif corruption_type == 'encoding':
                        affected_column = np.random.choice(task.categorical_columns)
                        random_corruptions.add(BrokenCharacters(affected_column, fraction))

            elif affected_column_type == 'text':

                if len(task.text_columns) >= 2 and np.random.uniform() < 0.1:
                    affected_columns = np.random.choice(task.text_columns, 2)
                    random_corruptions.add(SwappedValues(affected_columns[0], swap_with=affected_columns[1], fraction=fraction))

                else:
                    corruption_type = np.random.choice(['missing', 'encoding'])

                    if corruption_type == 'missing':
                        missingness = np.random.choice(['MCAR', 'MAR', 'MNAR'])
                        affected_column = np.random.choice(task.text_columns)
                        random_corruptions.add(MissingValues(affected_column, fraction, na_value='', missingness=missingness))

                    elif corruption_type == 'encoding':
                        affected_column = np.random.choice(task.text_columns)
                        random_corruptions.add(BrokenCharacters(affected_column, fraction))

        outcome = {
            'corruption': [],
            'status': [],
            'anomalies': [],
            'baseline_score': [],
            'corrupted_score': []
        }

        for corruption in random_corruptions:
            print(corruption)
            test_data_copy = task.test_data.copy(deep=True)
            corrupted_data = corruption.transform(test_data_copy)

            corrupted_data_stats = tfdv.generate_statistics_from_dataframe(corrupted_data)
            tfdv_anomalies = tfdv.validate_statistics(statistics=corrupted_data_stats, schema=schema)

            schema_anomalies = tfdv_anomalies.anomaly_info

            try:
                corrupted_predictions = model.predict_proba(corrupted_data)
                corrupted_score = task.score_on_test_data(corrupted_predictions)

                performance_drop = (baseline_score - corrupted_score) / baseline_score

                has_negative_impact = performance_drop > performance_threshold

            except Exception as err:
                corrupted_score = None
                has_negative_impact = True
                print(f"Raised error: {err}")

            has_anomalies = len(tfdv_anomalies.anomaly_info) != 0

            if has_anomalies:
                if has_negative_impact:
                    status = 'TP'

                else:
                    status = 'FP'

            else:
                if not has_negative_impact:
                    status = 'TN'

                else:
                    status = 'FN'

            outcome['corruption'].append(str(corruption))
            outcome['status'].append(status)
            outcome['anomalies'].append(str(schema_anomalies))
            outcome['baseline_score'].append(baseline_score)
            outcome['corrupted_score'].append(corrupted_score)

        return pd.DataFrame.from_dict(outcome)
