from jenga.tasks.income import IncomeEstimationTask
from jenga.corruptions.generic import MissingValues
from jenga.evaluation.corruption_impact import CorruptionImpactEvaluator

import numpy as np

task = IncomeEstimationTask(seed=42)

evaluator = CorruptionImpactEvaluator(task)

corruptions = []

for impacted_column in ['education', 'workclass']:
    for fraction in [0.99, 0.5, 0.01]:
        for missingness in ['MCAR', 'MAR', 'MNAR']:
            corruption = MissingValues(impacted_column, fraction, missingness=missingness, na_value='___')
            corruptions.append(corruption)


model = task.fit_baseline_model(task.train_data, task.train_labels)

results = evaluator.evaluate(model, 100, *corruptions)

for result in results:
    print(f"""
    Impact for {result.corruption}:
    ------------------------------------------
        Score on clean data:         {result.baseline_score}
        Scores on corrupted data:    {np.quantile(result.corrupted_scores, [0.1, 0.5, 0.9])}
    """)