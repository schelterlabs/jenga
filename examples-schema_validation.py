from jenga.tasks.income import IncomeEstimationTask
from jenga.corruptions.text import MissingValues
from jenga.evaluation.schema_validation import SchemaValidationEvaluator

import tensorflow_data_validation as tfdv

task = IncomeEstimationTask()

evaluator = SchemaValidationEvaluator(task)

schema = evaluator.schema_from_train_data()

education_feature = tfdv.get_feature(schema, 'education')
# Allow two percent unseen values in the education attribute
education_feature.distribution_constraints.min_domain_mass = 0.98

model = task.fit_baseline_model(task.train_data, task.train_labels)

many_missing_values = MissingValues('education', fraction=0.99, na_value='MISSING')
result = evaluator.evaluate_validation(model, schema, many_missing_values)[0]

print(f"""
Schema validation for {result.corruption}:
------------------------------------------
    Anomaly detected?           {result.anomalies_detected}
    Score on clean data:        {result.baseline_score}
    Score on corrupted data:    {result.corrupted_score}
    Performance drop in %:      {result.performance_drop_in_percent}
""")

few_missing_values = MissingValues('education', fraction=0.01, na_value='MISSING')
result = evaluator.evaluate_validation(model, schema, few_missing_values)[0]

print(f"""
Schema validation for {result.corruption}:
------------------------------------------
    Anomaly detected?           {result.anomalies_detected}
    Score on clean data:        {result.baseline_score}
    Score on corrupted data:    {result.corrupted_score}
    Performance drop in %:      {result.performance_drop_in_percent}
""")