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

from lime import lime_tabular

from ..corruptions.numerical import SwappedValues, Outliers, Scaling
from ..corruptions.categorical import SwapValues
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
                perturbation_fractions=[.5, .7, .9],
                verbose=False):
        self.pipeline = pipeline
        self.num_repetitions = num_repetitions
        self.perturbation_fractions = perturbation_fractions
        # assuming the first step is a ColumnTransformer with transformers named 
        # 'categorical_columns' or 'numerical_columns'
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.verbose = verbose
        
        self.perturbations = []
        for _ in range(self.num_repetitions):
            for fraction in self.perturbation_fractions:
                if len(self.numerical_columns)>1:
                    column_pairs = list(itertools.combinations(self.numerical_columns, 2))
                    swap_affected_column_pair = random.choice(column_pairs)
                    self.perturbations.append(('swapped', SwappedValues(fraction, swap_affected_column_pair)))
                    

                if len(self.categorical_columns)>1:
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
                    ('missing_MNAR', MissingValues(fraction, num_col, 0, 'MNAR')),
                    ('missing_high_entropy', MissingValuesHighEntropy(fraction, pipeline, [], [random.choice(self.numerical_columns)])),
                    ('missing_low_entropy', MissingValuesLowEntropy(fraction, pipeline, [], [random.choice(self.numerical_columns)]))
                    ]

                if self.categorical_columns:
                    cat_col = random.choice(self.categorical_columns)
                    self.perturbations += [
                    ('swap_categorical_values', SwapValues(fraction, cat_col)),
                    ('missing_MCAR', MissingValues(fraction, cat_col, '', 'MCAR')),
                    ('missing_MAR', MissingValues(fraction, cat_col, '', 'MAR')),
                    ('missing_MNAR', MissingValues(fraction, cat_col, '', 'MNAR')),
                    ('missing_high_entropy', MissingValuesHighEntropy(fraction, pipeline, [random.choice(self.categorical_columns)], [])),
                    ('missing_low_entropy', MissingValuesLowEntropy(fraction, pipeline, [random.choice(self.categorical_columns)], []))
                    ]

                if self.text_columns:
                    text_col = random.choice([self.text_columns])
                    self.perturbations.append(('broken_characters', BrokenCharacters(text_col, fraction)))

    @staticmethod
    def compute_ppp_features(predictions, bins_per_class_output=5):
        return np.percentile(predictions, 
                             np.arange(0, 101, bins_per_class_output),
                             axis=0).flatten()
    def _print(self, s):
        if self.verbose:
            print(s)

    def fit_ppp(self, X_df, y):

        self._print(f"Generating perturbed training data on {len(X_df)} rows ...")
        meta_features = []
        meta_scores = []
        for idx,perturbation in enumerate(self.perturbations):
            col = [v for k,v in perturbation[1].__dict__.items() if 'colum' in k][0]
            self._print(f'\t... perturbation {idx}/{len(self.perturbations)}: {perturbation[0]}, col {col}, fraction: {perturbation[1].fraction}')
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

        self._print("Training performance predictor...")
        self.meta_regressor = GridSearchCV(
                                meta_regressor_pipeline, 
                                param_grid, 
                                scoring='neg_mean_absolute_error')\
                                    .fit(meta_features, meta_scores)
        
        return self

    def predict_ppp(self, X_df):
        X_df[self.numerical_columns] = X_df[self.numerical_columns].fillna(0)
        meta_features = self.compute_ppp_features(self.pipeline.predict_proba(X_df))
        return self.meta_regressor.predict(meta_features.reshape(1, -1))[0]

    def predict_and_explain_ppp(self, 
                                X_df,
                                num_percentile_neighbors = 5,
                                num_top_meta_features = 3):
        # compute pipeline performance predictions with meta regressor
        pipeline_predictions = self.pipeline.predict_proba(X_df)
        meta_features = self.compute_ppp_features(pipeline_predictions)
        pipeline_performance_prediction = self.meta_regressor.predict(meta_features.reshape(1, -1))[0]
        
        # compute importances of percentiles (here: assuming random forest regressor
        # could be done with LIME or other explanability methods)
        rf = self.meta_regressor.best_estimator_.steps[-1][1]
        # find the top most important features, representing (percentile,pipeline-output-dim) tuples
        top_meta_feature_idx = rf.feature_importances_.argsort()[::-1][:num_top_meta_features]
        # get number of percentile bins (could also be stored in compute_ppp_features)
        num_perc_bins = int(meta_features.shape[0] / pipeline_predictions.shape[-1])

        relevant_samples = []
        # for each of the top features
        for important_feature_idx in top_meta_feature_idx:
            # get the relevant percentile
            important_percentile = meta_features[important_feature_idx]
            # find the original pipeline output dimension 
            original_feature_idx = int(np.floor(important_feature_idx / num_perc_bins))
            # find the datapoints closest to that percentile in that dimension
            percentile_neighbor_idx = np.abs((pipeline_predictions[:,original_feature_idx]-important_percentile)).argsort()[:num_percentile_neighbors]
            relevant_samples += percentile_neighbor_idx.tolist()

        return pipeline_performance_prediction, relevant_samples