import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from jenga.basis import BinaryClassificationTask


# Predict whether a person has high or low income based on demographic and financial attributes
class IncomeEstimationTask(BinaryClassificationTask):

    def __init__(self, seed, ignore_incomplete_records_for_training=False):

        columns = ['workclass', 'occupation', 'marital_status', 'education', 'hours_per_week', 'age']
        all_data = pd.read_csv('data/income/adult.csv', na_values='?')

        train_split, test_split = train_test_split(all_data, test_size=0.2)

        if ignore_incomplete_records_for_training:
            train_split = train_split.dropna()

        train_data = train_split[columns]
        train_labels = np.array(train_split['class'] == '>50K')

        test_data = test_split[columns]
        test_labels = np.array(test_split['class'] == '>50K')

        BinaryClassificationTask.__init__(self,
                                          seed,
                                          train_data,
                                          train_labels,
                                          test_data,
                                          test_labels,
                                          categorical_columns=['workclass', 'occupation', 'marital_status',
                                                               'education'],
                                          numerical_columns=['hours_per_week', 'age']
                                          )

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
            'learner__penalty': ['l2', 'l1', 'elasticnet'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))])

        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
        model = search.fit(train_data, train_labels)

        return model
