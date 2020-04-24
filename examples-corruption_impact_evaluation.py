from jenga.tasks.income import IncomeEstimationTask
from jenga.corruptions.text import MissingValues
from jenga.evaluation.corruption_impact import CorruptionImpactEvaluator

task = IncomeEstimationTask()

evaluator = CorruptionImpactEvaluator(task)

model = task.fit_baseline_model(task.train_data, task.train_labels)

corruptions = [
    MissingValues('education', fraction=0.99),
    MissingValues('education', fraction=0.5),
    MissingValues('education', fraction=0.01),
    MissingValues('workclass', fraction=0.99),
    MissingValues('workclass', fraction=0.01)
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
