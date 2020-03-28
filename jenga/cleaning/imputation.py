import pandas as pd

import autogluon as ag
from autogluon import TabularPrediction as task
import numpy as np
from typing import Any, List, Tuple, Dict

from datawig import SimpleImputer
from datawig.utils import set_stream_log_level

class Imputation:
    def __init__(self, categorical_columns, numeric_columns, text_columns=None):
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.text_columns = text_columns
   
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

class NoImputation(Imputation):
    def __call__(self, df):
        return df

class SimpleImputation(Imputation):
    def __call__(self, df):
        for c in self.categorical_columns:
            df.loc[df[c].isnull(),[c]] = df[c].dropna().value_counts().index[0]
        
        for c in self.numeric_columns:
            df.loc[df[c].isnull(),[c]] = df[c].dropna().median()
        return df


class AutoGluonImputation(Imputation):
    
    def __call__(self, 
                 df_orig: pd.DataFrame):
        # this is just to prevent autogluon from raising type errors
        df = df_orig.copy(deep=True)
        for col in df.columns:
            if pd.api.types.is_categorical_dtype(df[col]):
                df[col] = df[col].astype(str)

        observed = df.isnull() == False

        for col in self.categorical_columns + self.numeric_columns:
            if observed[col].sum() < len(df):
                train_df = df[observed[col]]
                test_df = df.loc[observed[col] == False, :].drop(col, axis=1)
                predictor = task.fit(train_data=train_df, label=col)
                df.loc[observed[col] == False, col] = predictor.predict(test_df)
        return df

class DatawigImputation(Imputation):
    def __call__(self, df):
        for c in self.categorical_columns + self.numeric_columns:
            input_cols = list(set(df.columns) - set([c]))
            imputer = SimpleImputer(input_columns=input_cols, output_column=c)
            set_stream_log_level("ERROR")
            missing = df[c].isnull()
            if missing.sum()>0:
                imputer = imputer.fit(df.loc[~missing, :])
                print(f"Imputing {missing.sum()} missing values in column {c}")
                df_tmp = imputer.predict(df)
                df[c] = df_tmp[c + "_imputed"]
            else:
                print(f"no missing values detected in column {c}")
        return df