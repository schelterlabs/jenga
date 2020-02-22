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
from .. import corruptions
# from corruptions.numerical import SwappedValues, Outliers, Scaling
# from corruptions.text import BrokenCharacters
# from corruptions.missing import ( MissingValuesHighEntropy, 
#                                   MissingValuesLowEntropy, 
#                                   MissingValues
#                                 )

class PipelineWithPPP:

    def __init__(self, 
                pipeline, 
                numerical_columns = [],
                categorical_columns = [],
                text_columns = [],
                num_repetitions=10, 
                perturbation_fractions=[.1, .2, .5, .9]):
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
                self.perturbations.append(('swapped', corruptions.numerical.SwappedValues(fraction, swap_affected_column_pair)))
                
                column_pairs = list(itertools.combinations(self.categorical_columns, 2))
                swap_affected_column_pair = random.choice(column_pairs)
                self.perturbations.append(('swapped', corruptions.numerical.SwappedValues(fraction, swap_affected_column_pair)))

                num_col = random.choice(self.numerical_columns)
            
                self.perturbations += [
                ('scaling', corruptions.numerical.Scaling(fraction, [num_col])),
                ('outlier', corruptions.numerical.Outliers(fraction, [num_col])),
                ('missing_MCAR', corruptions.missing.MissingValues(fraction, num_col, 0, 'MCAR')),
                ('missing_MAR', corruptions.missing.MissingValues(fraction, num_col, 0, 'MAR')),
                ('missing_MNAR', corruptions.missing.MissingValues(fraction, num_col, 0, 'MNAR'))
                ]
            
                cat_col = random.choice(self.categorical_columns)
                self.perturbations += [
                ('missing_MCAR', corruptions.missing.MissingValues(fraction, cat_col, '', 'MCAR')),
                ('missing_MAR', corruptions.missing.MissingValues(fraction, cat_col, '', 'MAR')),
                ('missing_MNAR', corruptions.missing.MissingValues(fraction, cat_col, '', 'MNAR'))
                ]

                self.perturbations += [
                    ('missing_high_entropy', corruptions.missing.MissingValuesHighEntropy(fraction, pipeline, [cat_col], [num_col])),
                    ('missing_low_entropy', corruptions.missing.MissingValuesLowEntropy(fraction, pipeline, [cat_col], [num_col]))
                ]
                
                text_col = random.choice(self.text_columns)
                self.perturbations.append(('broken_characters', 
                            corruptions.text.BrokenCharacters(text_col, fraction)))

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

        print("Generating perturbed training data...")
        meta_features = []
        meta_scores = []
        for perturbation in self.perturbations:
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