from jenga.tasks.reviews import VideogameReviewsTask
from jenga.corruptions.generic import MissingValues
from jenga.corruptions.numerical import GaussianNoise, Scaling
from jenga.corruptions.text import BrokenCharacters
from jenga.evaluation.schema_validation import SchemaValidationEvaluator

import tensorflow_data_validation as tfdv

task = VideogameReviewsTask(seed=42)

evaluator = SchemaValidationEvaluator(task)

schema = evaluator.schema_from_train_data()

#print(schema)
review_date_feature = tfdv.get_feature(schema, 'review_date')
review_date_feature.distribution_constraints.min_domain_mass = 0.0


#education_feature = tfdv.get_feature(schema, 'education')
# Allow two percent unseen values in the education attribute
#education_feature.distribution_constraints.min_domain_mass = 0.98
#education_feature.presence.min_fraction = 0.90#distribution_constraints.min_domain_mass = 0.98

#print(schema)

#model = task.fit_baseline_model(task.train_data, task.train_labels)

import pickle
#with open('psv.pkl', 'wb') as output:
#   pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)

with open('psv.pkl', 'rb') as f:
    model = pickle.load(f)

corruptions = [
    MissingValues('title_and_review_text', fraction=0.25, na_value=''),
    MissingValues('title_and_review_text', fraction=0.05, na_value=''),
    MissingValues('title_and_review_text', fraction=0.01, na_value=''),
    MissingValues('title_and_review_text', fraction=0.005, na_value=''),
    MissingValues('verified_purchase', fraction=0.25, na_value=''),
    MissingValues('verified_purchase', fraction=0.05, na_value=''),
    MissingValues('verified_purchase', fraction=0.01, na_value=''),
    MissingValues('verified_purchase', fraction=0.005, na_value=''),
    MissingValues('vine', fraction=0.25, na_value=''),
    MissingValues('vine', fraction=0.05, na_value=''),
    MissingValues('vine', fraction=0.01, na_value=''),
    MissingValues('vine', fraction=0.005, na_value=''),
    BrokenCharacters('title_and_review_text', fraction=0.25),
    BrokenCharacters('title_and_review_text', fraction=0.05),
    BrokenCharacters('title_and_review_text', fraction=0.01),
    BrokenCharacters('title_and_review_text', fraction=0.005),
    GaussianNoise('star_rating', fraction=0.25),
    GaussianNoise('star_rating', fraction=0.05),
    GaussianNoise('star_rating', fraction=0.01),
    GaussianNoise('star_rating', fraction=0.005),
    Scaling('star_rating', fraction=0.25),
    Scaling('star_rating', fraction=0.05),
    Scaling('star_rating', fraction=0.01),
    Scaling('star_rating', fraction=0.005),
]

results = evaluator.evaluate_validation(model, schema, 10, *corruptions)

for result in results:
    print(f"""
    Schema validation for {result.corruption}:
    ------------------------------------------
        Anomaly detected?           {result.anomalies_detected}
        Score on clean data:        {result.baseline_score}
        Scores on corrupted data:   {result.corrupted_scores}
    """)


import jsonpickle
with open("schemaval-results.jsonpickle", "w") as text_file:
    text_file.write(jsonpickle.encode(results))


