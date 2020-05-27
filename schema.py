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

print(schema)
