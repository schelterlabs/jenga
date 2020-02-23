import pandas as pd
import numpy as np
import random
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

from ..corruptions.numerical import SwappedValues, Outliers, Scaling
from ..corruptions.text import BrokenCharacters
from ..corruptions.missing import ( MissingValuesHighEntropy, 
                                  MissingValuesLowEntropy, 
                                  MissingValues
                                )

class PipelineWithPPP:

    def __init__(self, 
                pipeline, 
                numerical_columns = [],
                categorical_columns = [],
                text_columns = [],
                num_repetitions=5, 
                perturbation_fractions=[.5, .7, .9]):
        self.pipeline = pipeline
        self.num_repetitions = num_repetitions
        self.perturbation_fractions = perturbation_fractions
        # assuming the first step is a ColumnTransformer with transformers named 
        # 'categorical_columns' or 'numerical_columns'
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        
        self.perturbations = []
        for _ in range(self.num_repetitions):
            for fraction in self.perturbation_fractions:
                column_pairs = list(itertools.combinations(self.numerical_columns, 2))
                swap_affected_column_pair = random.choice(column_pairs)
                self.perturbations.append(('swapped', SwappedValues(fraction, swap_affected_column_pair)))
                
                column_pairs = list(itertools.combinations(self.categorical_columns, 2))
                swap_affected_column_pair = random.choice(column_pairs)
                self.perturbations.append(('swapped', SwappedValues(fraction, swap_affected_column_pair)))

                if self.numerical_columns:
                    num_col = random.choice(self.numerical_columns)
                
                    self.perturbations += [
                    ('scaling', Scaling(fraction, [num_col])),
                    ('outlier', Outliers(fraction, [num_col])),
                    ('missing_MCAR', MissingValues(fraction, num_col, 0, 'MCAR')),
                    ('missing_MAR', MissingValues(fraction, num_col, 0, 'MAR')),
                    ('missing_MNAR', MissingValues(fraction, num_col, 0, 'MNAR'))
                    ]
                if self.categorical_columns:
                    cat_col = random.choice(self.categorical_columns)
                    self.perturbations += [
                    ('missing_MCAR', MissingValues(fraction, cat_col, '', 'MCAR')),
                    ('missing_MAR', MissingValues(fraction, cat_col, '', 'MAR')),
                    ('missing_MNAR', MissingValues(fraction, cat_col, '', 'MNAR'))
                    ]

                if self.categorical_columns or self.numerical_columns:
                    self.perturbations += [
                        ('missing_high_entropy', MissingValuesHighEntropy(fraction, pipeline, [random.choice(self.categorical_columns)], [random.choice(self.numerical_columns)])),
                        ('missing_low_entropy', MissingValuesLowEntropy(fraction, pipeline, [random.choice(self.categorical_columns)], [random.choice(self.numerical_columns)]))
                    ]
                    
                if self.text_columns:
                    text_col = random.choice([self.text_columns])
                    self.perturbations.append(('broken_characters', BrokenCharacters(text_col, fraction)))

    @staticmethod
    def compute_ppp_features(predictions):
        probs_class_a = np.transpose(predictions)[0]
        features_a = np.percentile(probs_class_a, np.arange(0, 101, 5))
        if predictions.shape[-1] > 1:
            probs_class_b = np.transpose(predictions)[1]
            features_b = np.percentile(probs_class_b, np.arange(0, 101, 5))
            return np.concatenate((features_a, features_b), axis=0)
        else:
            return features_a

    def fit_ppp(self, X_df, y):

        print(f"Generating perturbed training data on {len(X_df)} rows ...")
        meta_features = []
        meta_scores = []
        for idx,perturbation in enumerate(self.perturbations):
            col = [v for k,v in perturbation[1].__dict__.items() if 'colum' in k][0]
            print(f'\t... perturbation {idx}/{len(self.perturbations)}: {perturbation[0]}, col {col}, fraction: {perturbation[1].fraction}')
            df_perturbed = perturbation[1](X_df)
            predictions = self.pipeline.predict_proba(df_perturbed)
            meta_features.append(self.compute_ppp_features(predictions))
            meta_scores.append(self.pipeline.score(df_perturbed, y))
 
        param_grid = {
            'learner__n_estimators': np.arange(5, 20, 5),
            'learner__criterion': ['mae']
        }

        meta_regressor_pipeline = Pipeline([
           ('scaling', StandardScaler()),
           ('learner', RandomForestRegressor(criterion='mae'))
        ])

        print("Training performance predictor...")
        self.meta_regressor = GridSearchCV(
                                meta_regressor_pipeline, 
                                param_grid, 
                                scoring='neg_mean_absolute_error')\
                                    .fit(meta_features, meta_scores)
        
        return self

    def predict_ppp(self, X_df):
        meta_features = self.compute_ppp_features(self.pipeline.predict_proba(X_df))
        return self.meta_regressor.predict(meta_features.reshape(1, -1))[0]