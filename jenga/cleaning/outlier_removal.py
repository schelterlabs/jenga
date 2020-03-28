import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from pyod.models.knn import KNN
from autogluon import TabularPrediction as task

class OutlierRemoval:
    def __init__(self, train_df, categorical_columns, numeric_columns, text_columns):

        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.text_columns = text_columns
    
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        transformers = [
            ('numeric_features', numeric_transformer, self.numeric_columns),
            ('categorical_features', categorical_transformer, self.categorical_columns)
        ]

        if self.text_columns:
            transformers.append(('textual_features', 
                                  HashingVectorizer(ngram_range=(1, 3), 
                                                         n_features=2**15), 
                                  ['PyOD_outlier_text_dummy_column']))
            
        self.feature_transformation = ColumnTransformer(transformers=transformers, 
                                                    sparse_threshold=1.0)
        
        if self.text_columns:
            train_df['OutlierRemoval_text_dummy_column'] = ''
            for text_column in self.text_columns:
                train_df['OutlierRemoval_text_dummy_column'] += " " + train_df[text_column]

        self.feature_transformation = self.feature_transformation.fit(train_df)
        X = self.feature_transformation.transform(train_df)
        self.model = IsolationForest(random_state=0, contamination=.1).fit(X)

        if self.text_columns:
            train_df = train_df.drop(['OutlierRemoval_text_dummy_column'], axis=1)
     
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

class NoOutlierRemoval(OutlierRemoval):
     def __call__(self, df):
        return df

class PyODKNN(OutlierRemoval):
    def __call__(self, df):

        if self.text_columns:
            df['OutlierRemoval_text_dummy_column'] = ''
            for text_column in self.text_columns:
                df['OutlierRemoval_text_dummy_column'] += " " + df[text_column]

        X = self.feature_transformation.fit_transform(df).to_array()

        df['outlier_score'] = self.model.predict(X)
        
        if self.text_columns:
            df = df.drop(['OutlierRemoval_text_dummy_column'], axis=1)
        
        return df

class SKLearnIsolationForest(OutlierRemoval):
    def __call__(self, df):

        if self.text_columns:
            df['OutlierRemoval_text_dummy_column'] = ''
            for text_column in self.text_columns:
                df['OutlierRemoval_text_dummy_column'] += " " + df[text_column]

        X = self.feature_transformation.fit_transform(df)
        df['outlier_score'] = self.model.fit_predict(X) < 0

        if self.text_columns:
            df = df.drop(['OutlierRemoval_text_dummy_column'], axis=1)
        
        return df


class AutoGluonCleanerNoTrainingData(OutlierRemoval):
    
    def __init__(self, 
                numeric_columns,
                categorical_columns,
                text_columns,
                categorical_precision_threshold=.9, 
                numerical_std_error_threshold=3,
                cv_folds = 2):
        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold
        self.cv_folds = cv_folds

    def __call__(self, df_orig: pd.DataFrame):

        # this is just to prevent autogluon from raising type errors
        df = df_orig.copy(deep=True)
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)

        folds = np.random.permutation(np.arange(len(df)) % self.cv_folds)
        for fold in range(self.cv_folds):
            
            train_df = task.Dataset(df[folds == fold])
            test_df = task.Dataset(df[folds != fold])

            for col in df.columns:
                predictor = task.fit(train_data=train_df, label=col)
                
                y_pred = predictor.predict(test_df)
                y_test = test_df[col] 
                perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
                if 'root_mean_squared_error' in perf:
                    idx_too_wrong = np.sqrt((y_pred - y_test)**2) > perf['root_mean_squared_error'] * self.numerical_std_error_threshold
                else:
                    not_interesting = ['accuracy', 'macro avg', 'weighted avg']
                    labels = [k for k in perf['<lambda>'].keys() if k not in not_interesting]
                    idx_too_wrong = False
                    for label in labels:
                        if perf['<lambda>'][label]['precision'] > self.categorical_precision_threshold:
                            idx_too_wrong = idx_too_wrong | (df[col]==label)
                idx = df[col].isnull() | idx_too_wrong
                df.loc[idx, col] = y_pred[idx]
                    
        return df
   
class AutoGluonCleaner(OutlierRemoval):
    
    def __init__(self,
                train_df,
                numeric_columns=None,
                categorical_columns=None,
                text_columns=None,
                categorical_precision_threshold=.85, 
                numerical_std_error_threshold=1):
        self.categorical_precision_threshold = categorical_precision_threshold
        self.numerical_std_error_threshold = numerical_std_error_threshold
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        # this is just to prevent autogluon from raising type errors
        df = train_df.copy(deep=True)
        
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)
        
        self.predictors = {}
        for col in self.categorical_columns:
            if len(df[col].unique()) < len(df):
                self.predictors[col] = task.fit(train_data=df, label=col, problem_type='multiclass')
        for col in self.numeric_columns:
            self.predictors[col] = task.fit(train_data=df, label=col, problem_type='regression')

    def __call__(self, df_orig: pd.DataFrame):

        # this is just to prevent autogluon from raising type errors
        df = df_orig.copy(deep=True)
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)

        for col in self.categorical_columns + self.numeric_columns:
            y_pred = self.predictors[col].predict(df)
            y_test = df[col] 
            valid = y_test.isnull()==False
            perf = self.predictors[col].evaluate_predictions(y_true=y_test[valid], y_pred=y_pred[valid], auxiliary_metrics=True)
            if 'root_mean_squared_error' in perf:
                idx_too_wrong = np.sqrt((y_pred - y_test.fillna(0))**2) > perf['root_mean_squared_error'] * self.numerical_std_error_threshold
            else:
                not_interesting = ['accuracy', 'macro avg', 'weighted avg']
                labels = [k for k in perf['<lambda>'].keys() if k not in not_interesting]
                idx_too_wrong = False
                for label in labels:
                    if perf['<lambda>'][label]['precision'] > self.categorical_precision_threshold:
                        idx_too_wrong = idx_too_wrong | (df[col]==label)
            idx = df[col].isnull() | idx_too_wrong
            df.loc[idx, col] = y_pred[idx]
                    
        return df
   
