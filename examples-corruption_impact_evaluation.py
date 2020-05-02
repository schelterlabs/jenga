from jenga.tasks.income import IncomeEstimationTask
from jenga.corruptions.generic import MissingValues
from jenga.evaluation.corruption_impact import CorruptionImpactEvaluator

task = IncomeEstimationTask(seed=42)

evaluator = CorruptionImpactEvaluator(task)

model = task.fit_baseline_model(task.train_data, task.train_labels)

corruptions = [
    MissingValues('education', fraction=0.99, na_value=''),
    MissingValues('education', fraction=0.5, na_value=''),
    MissingValues('education', fraction=0.01, na_value=''),
    MissingValues('workclass', fraction=0.99, na_value=''),
    MissingValues('workclass', fraction=0.01, na_value='')
]

results = evaluator.evaluate(model, *corruptions)

for result in results:
    print(f"""
    Impact for {result.corruption}:
    ------------------------------------------
        Score on clean data:        {result.baseline_score}
        Score on corrupted data:    {result.corrupted_score}
        Performance drop in %:      {result.performance_drop_in_percent}
    """)
