import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Cleaner:
    def __init__(self,
                train_df: pd.DataFrame,
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

    def fit(self, df, columns=None):
        pass

    def remove_outliers(self, df, columns=None):
        pass

    def impute(self, df, columns=None):
        pass

    def clean(self, df, columns=None):
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
            y_test = test_df[col].dropna(subset=[col])
            y_pred = self.predictors[col].predict(test_df.drop(col,axis=1).dropna(subset=[col]))
            perf = self.predictors[col].evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
            not_interesting = ['accuracy', 'macro avg', 'weighted avg']
            labels = [k for k in perf['<lambda>'].keys() if k not in not_interesting]
            high_precision_labels = []
            for label in labels:
                if perf['<lambda>'][label]['precision'] > self.categorical_precision_threshold:
                    high_precision_labels += label
            if high_precision_labels:
                self.predictable_columns += col
                self.predictors[col].high_precision_labels = high_precision_labels

        for col in self.numeric_columns:
            self.predictors[col] = task.fit(train_data=train_df.dropna(subset=[col]), 
                                            label=col, 
                                            problem_type='regression', 
                                            verbosity=0)
            y_test = test_df[col].dropna(subset=[col])
            y_pred = self.predictors[col].predict(test_df.drop(col,axis=1).dropna(subset=[col]))
            perf = self.predictors[col].evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
            if perf['root_mean_squared_error'] < self.numerical_std_error_threshold * y_test.std():
                self.predictable_columns += col
        print(f'Found {len(self.predictable_columns)}: {self.predictable_columns}')
            
    def remove_outliers(self, df: pd.DataFrame, columns=None):
        
        if not columns:
            columns = self.categorical_columns + self.numeric_columns

        df = self._categorical_columns_to_string(df.copy(deep=True))

        for col in self.predictable_columns:
            y_pred = self.predictors[col].predict(df)
            num_nans = df[col].isnull().sum()
            if col in self.categorical_columns:
                presumably_wrong = y_pred.isin(self.predictors[col].high_precision_labels) & df[col] != y_pred
                    
            if col in self.numeric_columns:
                presumably_wrong = np.sqrt((y_pred - y_test.fillna(np.inf))**2) > perf['root_mean_squared_error'] * self.numerical_std_error_threshold
            
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

    def clean(self, df: pd.DataFrame, columns=None):

        df = self.remove_outliers(df, columns)
        df = self.impute(df, columns)
        
        return df
        