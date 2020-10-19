import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from autogluon import TabularPrediction as task
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

class Cleaner:
    def __init__(self,
                numeric_columns=None,
                categorical_columns=None,
                text_columns=None,
                categorical_precision_threshold=.85, 
                numerical_std_error_threshold=1.,
                test_size=.2):
        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.test_size=test_size
        self.predictors = {}
        self.predictable_columns = []
    
    @staticmethod
    def _categorical_columns_to_string(df):
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)
        return df

    def fit(self, df, columns=None):
        pass

    def remove_outliers(self, df, columns=None):
        pass

    def impute(self, df, columns=None):
        pass

    def clean(self, df: pd.DataFrame, columns=None):

        df = self.remove_outliers(df, columns)
        df = self.impute(df, columns)
        
        return df
        

class DatawigCleaner:
    def __init__(self,
                numeric_columns=None,
                categorical_columns=None,
                text_columns=None,
                categorical_precision_threshold=.85, 
                numerical_std_error_threshold=1.,
                test_size=.2):
        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.test_size=test_size
        self.predictors = {}
        self.predictable_columns = []

    def fit(self, df, columns=None):
        
        if not columns:
            columns = self.categorical_columns + self.numeric_columns

        df = self._categorical_columns_to_string(df)
        
        train_df, test_df = train_test_split(df, test_size=self.test_size)

        for col in self.categorical_columns:
            self.predictors[col] = datawig.SimpleImputer(
            input_columns=list(set(columns)-set(col)), # column(s) containing information about the column we want to impute
            output_column=col, # the column we'd like to impute values for
            output_path = 'imputer_model' # stores model data and metrics
            )

    def remove_outliers(self, df, columns=None):
        pass

    def impute(self, df, columns=None):
        pass


class SklearnCleaner(Cleaner):
    def __init__(self,
                numeric_columns=None,
                categorical_columns=None,
                text_columns=None,
                categorical_precision_threshold=.85, 
                numeric_error_percentile=.9,
                test_size=.2):

        self.categorical_precision_threshold = categorical_precision_threshold
        self.numeric_error_percentile = numeric_error_percentile
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.test_size=test_size
        self.predictors = {}
        self.predictable_columns = []

    def fit(self, df, columns=None):
        if not columns:
            columns = self.categorical_columns + self.numeric_columns

        df = self._categorical_columns_to_string(df)

        

        train_df, test_df = train_test_split(df, test_size=self.test_size)

        for col in columns:
            categorical_preprocessing = Pipeline([
            ('mark_missing', SimpleImputer(strategy='constant', fill_value='__NA__')),
            ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
            ])

            numeric_preprocessing = Pipeline([
                ('mark_missing', SimpleImputer(strategy='median')),
                ('scaling',  StandardScaler())
            ])
            if col in self.categorical_columns:
                feature_transformation = ColumnTransformer(transformers=[
                    ('categorical_features', categorical_preprocessing, list(set(self.categorical_columns)-{col})),
                    ('scaled_numeric', numeric_preprocessing, self.numeric_columns)
                ])

                param_grid = {
                    'learner__n_estimators': [100, 200],
                }

                pipeline = Pipeline([
                    ('features', feature_transformation),
                    ('learner', GradientBoostingClassifier())])
                search = GridSearchCV(pipeline, param_grid, cv=2, verbose=0, n_jobs=-1)
                self.predictors[col] = search.fit(train_df, train_df[col])
                print(f'Classifer for column {col} reached {search.best_score_}')

                # prec-rec curves for finding the likelihood thresholds for minimal precision
                self.predictors[col].thresholds = {}
                probas = self.predictors[col].predict_proba(test_df)
                for label_idx, label in enumerate(self.predictors[col].classes_):
                    prec, rec, threshold = precision_recall_curve(test_df[col]==label, probas[:,label_idx], pos_label=True)
                    threshold_for_minimal_precision = threshold[(prec >= self.categorical_precision_threshold).nonzero()[0][0]]
                    self.predictors[col].thresholds[label] = threshold_for_minimal_precision


            elif col in self.numeric_columns:
                feature_transformation = ColumnTransformer(transformers=[
                    ('categorical_features', categorical_preprocessing, self.categorical_columns),
                    ('scaled_numeric', numeric_preprocessing, list(set(self.numeric_columns)-{col}))
                ])

                param_grid = {
                    'learner__n_estimators': [10, 100],
                }

                self.predictors[col] = {}
                for perc_name, percentile in zip(['lower','median','upper'],
                    [1.-self.numeric_error_percentile, .5, self.numeric_error_percentile]):
                    pipeline = Pipeline([
                        ('features', feature_transformation),
                        ('learner', GradientBoostingRegressor(loss='quantile',alpha=percentile))])

                    search = GridSearchCV(pipeline, param_grid, cv=2, verbose=0, n_jobs=-1)
                    self.predictors[col][perc_name] = search.fit(train_df, train_df[col])
                    print(f'Regressor for column {col}/{perc_name} reached {search.best_score_}')
        return self

    def remove_outliers(self, df, columns=None):
        if not columns:
            columns = self.categorical_columns + self.numeric_columns

        df = self._categorical_columns_to_string(df.copy(deep=True))

        for col in columns:
            
            if col in self.categorical_columns:
                y_pred = self.predictors[col].predict(df)    
                y_proba = self.predictors[col].predict_proba(df)    
                for label_idx, label in enumerate(self.predictors[col].classes_):
                    import pdb;pdb.set_trace()
                    above_prec_predictions = self.predictors[col].thresholds[label] <= y_proba[:,label_idx]
                    outliers = above_prec_predictions & (df[col] != y_pred)

            if col in self.numeric_columns:
                lower_percentile = self.predictors[col]['lower'].predict(df)
                upper_percentile = self.predictors[col]['upper'].predict(df)
                outliers = (df[col] < lower_percentile) | (df[col] > upper_percentile)
            
            num_nans = df[col].isnull().sum()
            df.loc[outliers, col] = np.nan

            print(f'Column {col} contained {num_nans} nans before, now {df[col].isnull().sum()}')
        
        return df

    def impute(self, df, columns=None):
        if not columns:
            columns = self.categorical_columns + self.numeric_columns
        
        df = self._categorical_columns_to_string(df.copy(deep=True))

        for col in columns:
            prior_missing = df[col].isnull().sum()
            if prior_missing > 0:
                if col in self.numeric_columns:
                    df.loc[df[col].isnull(), col] = self.predictors[col]['median'].predict(df[df[col].isnull()])
                elif col in self.categorical_columns:
                    df.loc[df[col].isnull(), col] = self.predictors[col].predict(df[df[col].isnull()])
                print(f'Imputed {prior_missing} values in column {col}')

        return df

class AutoencoderCleaner:
    def __init__(self,
                numeric_columns=None,
                categorical_columns=None,
                text_columns=None,
                categorical_precision_threshold=.85, 
                numerical_std_error_threshold=1.,
                test_size=.2):
        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.test_size=test_size
        self.predictors = {}
        self.predictable_columns = []

    def fit(self, df, columns=None):
        pass

    def remove_outliers(self, df, columns=None):
        pass

    def impute(self, df, columns=None):
        pass

class AutoGluonCleaner(Cleaner):

    def fit(self, df, columns=None):
        
        if not columns:
            columns = self.categorical_columns + self.numeric_columns

        df = self._categorical_columns_to_string(df)
        
        train_df, test_df = train_test_split(df, test_size=self.test_size)

        for col in self.categorical_columns:
            self.predictors[col] = task.fit(train_data=train_df.dropna(subset=[col]), 
                                                label=col, 
                                                problem_type='multiclass', 
                                                verbosity=0)
            y_test = test_df[col].dropna()
            y_pred = self.predictors[col].predict(test_df.dropna(subset=[col]).drop([col],axis=1))
            perf = self.predictors[col].evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
            not_interesting = ['accuracy', 'macro avg', 'weighted avg']
            labels = [k for k in perf['classification_report'].keys() if k not in not_interesting]
            high_precision_labels = []
            for label in labels:
                if perf['classification_report'][label]['precision'] > self.categorical_precision_threshold:
                    high_precision_labels += label
            if high_precision_labels:
                self.predictable_columns.append(col)
                self.predictors[col].high_precision_labels = high_precision_labels

        for col in self.numeric_columns:
            self.predictors[col] = task.fit(train_data=train_df.dropna(subset=[col]), 
                                            label=col, 
                                            problem_type='regression', 
                                            verbosity=0)
            y_test = test_df[col].dropna()
            y_pred = self.predictors[col].predict(test_df.dropna(subset=[col]).drop([col],axis=1))
            perf = self.predictors[col].evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
            if perf['root_mean_squared_error'] < self.numerical_std_error_threshold * y_test.std():
                self.predictable_columns.append(col)
                self.predictors[col].root_mean_squared_error = perf['root_mean_squared_error']

        print(f'Found {len(self.predictable_columns)} predictable columns: {self.predictable_columns}')
            
    def remove_outliers(self, df: pd.DataFrame, columns=None):
        
        if not columns:
            columns = self.categorical_columns + self.numeric_columns

        df = self._categorical_columns_to_string(df.copy(deep=True))

        for col in self.predictable_columns:
            y_pred = self.predictors[col].predict(df)
            y_test = df[col]
            num_nans = df[col].isnull().sum()
            if col in self.categorical_columns:
                presumably_wrong = y_pred.isin(self.predictors[col].high_precision_labels) & df[col] != y_pred
                    
            if col in self.numeric_columns:
                rmse = self.predictors[col].root_mean_squared_error
                presumably_wrong = np.sqrt((y_pred - y_test.fillna(np.inf))**2) > rmse * self.numerical_std_error_threshold
            
            df.loc[presumably_wrong, col] = np.nan

            print(f'Column {col} contained {num_nans} nans before, now {df[col].isnull().sum()}')
        
        return df

    def impute(self, df, columns=None):
        
        if not columns:
            columns = self.categorical_columns + self.numeric_columns
        df = self._categorical_columns_to_string(df.copy(deep=True))

        for col in self.predictable_columns:
            df.loc[df[col].isnull(), col] = self.predictors[col].predict(df[df[col].isnull()])
            print(f'Imputed {df[col].isnull().sum()} values in column {col}')

        return df

    