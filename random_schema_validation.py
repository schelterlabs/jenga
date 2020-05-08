from jenga.tasks.reviews import VideogameReviewsTask
from jenga.corruptions.generic import MissingValues, SwappedValues
from jenga.corruptions.numerical import GaussianNoise, Scaling
from jenga.corruptions.text import BrokenCharacters

import tensorflow_data_validation as tfdv

import numpy as np
import pandas as pd

seed = np.random.randint(2**32-1)

print(seed)

task = VideogameReviewsTask(seed=seed)

train_data_stats = tfdv.generate_statistics_from_dataframe(task.train_data)
schema = tfdv.infer_schema(statistics=train_data_stats)
review_date_feature = tfdv.get_feature(schema, 'review_date')
review_date_feature.distribution_constraints.min_domain_mass = 0.0


def validate(schema, data):
    stats = tfdv.generate_statistics_from_dataframe(data)
    return tfdv.validate_statistics(statistics=stats, schema=schema)


def has_anomaly(validation_result):
    return len(validation_result.anomaly_info) != 0


# Make sure the schema works on the test data
assert(not has_anomaly(validate(schema, task.test_data)))

import pickle
with open('psv.pkl', 'rb') as f:
    model = pickle.load(f)

baseline_predictions = model.predict_proba(task.test_data)
baseline_score = task.score_on_test_data(baseline_predictions)

num_repetitions = 250

threshold = 0.02

random_corruptions = set()

for _ in range(0, num_repetitions):
    fraction = float(np.random.randint(100)) / 100
    corruption_type = np.random.choice(['missing', 'swapped', 'noise', 'scaling', 'encoding'])

    if corruption_type == 'missing':
        missingness = np.random.choice(['MCAR', 'MAR', 'MNAR'])
        if np.random.uniform() < 0.5:
            affected_column = np.random.choice(task.categorical_columns + task.text_columns)
            random_corruptions.add(MissingValues(affected_column, fraction, na_value=''))
        else:
            affected_column = np.random.choice(task.numerical_columns)
            random_corruptions.add(MissingValues(affected_column, fraction, na_value=np.nan))
    elif corruption_type == 'swapped':
        affected_columns = np.random.choice(task.categorical_columns + task.text_columns, 2)
        random_corruptions.add(SwappedValues(affected_columns[0], affected_columns[1], fraction))
    elif corruption_type == 'noise':
        affected_column = np.random.choice(task.numerical_columns)
        random_corruptions.add(GaussianNoise(affected_column, fraction))
    elif corruption_type == 'scaling':
        affected_column = np.random.choice(task.numerical_columns)
        random_corruptions.add(Scaling(affected_column, fraction))
    elif corruption_type == 'encoding':
        affected_column = np.random.choice(task.categorical_columns + task.text_columns)
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

        has_negative_impact = performance_drop > threshold
    except:
        corrupted_score = None
        has_negative_impact = True

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

outcome_df = pd.DataFrame.from_dict(outcome)

print(outcome_df.to_string())

outcome_df.to_csv(f'random_validation_{seed}.csv', index=False)
