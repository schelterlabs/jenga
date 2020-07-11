import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml

from jenga.cleaning.Cleaners import AutoGluonCleaner
from jenga.basis import BinaryClassificationTask

# some binary classification tasks from 
# https://www.openml.org/search?q=qualities.NumberOfClasses%3A2%2520qualities.NumberOfInstances%3Alt%3B10000%2520qualities.NumberOfFeatures%3Alt%3B100&type=data
OPENML_IDS = [346, 1448, 40994]

# Predict whether a person has high or low income based on demographic and financial attributes
class AutoCleaningTask(BinaryClassificationTask):

    def __init__(self, seed, cleaner=AutoGluonCleaner):

        X, y = fetch_openml(data_id=OPENML_IDS[1], as_frame=True, return_X_y=True)

        categorical_columns, numeric_columns = self._guess_dtypes(X)
        labels = y.unique()
        
        train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

        BinaryClassificationTask.__init__(self,
                                          seed,
                                          train_data,
                                          train_labels,
                                          test_data,
                                          test_labels,
                                          categorical_columns=categorical_columns,
                                          numerical_columns=numeric_columns
                                          )
        self.cleaner = cleaner(  
                                numeric_columns=numeric_columns, 
                                categorical_columns=categorical_columns,
                                categorical_precision_threshold=.85, 
                                numerical_std_error_threshold=1.
                        )
        

    def _is_categorical(self, col, max_unique_ratio = 0.05):
        return len(col.value_counts()) / len(col) < max_unique_ratio 

    def _guess_dtypes(self, df):
        categorical_columns = [c for c in df.columns if self._is_categorical(df[c])]
        numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) \
                           and c not in categorical_columns]
        return categorical_columns, numeric_columns

    def fit_cleaner(self, train_data):
        self.cleaner.fit(train_data)

    def fit_baseline_model(self, train_data, train_labels):

        mark_missing_and_encode = Pipeline([
            ('mark_missing', SimpleImputer(strategy='constant', fill_value='__NA__')),
            ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
        ])

        feature_transformation = ColumnTransformer(transformers=[
            ('categorical_features', mark_missing_and_encode, self.categorical_columns),
            ('scaled_numeric', StandardScaler(), self.numerical_columns)
        ])

        param_grid = {
            'learner__loss': ['log'],
            'learner__alpha': [0.0001, 0.001, 0.01]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))])

        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
        model = search.fit(train_data, train_labels)

        return model
