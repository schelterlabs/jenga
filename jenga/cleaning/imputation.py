import pandas as pd
from datawig import SimpleImputer

class Imputation:
    def __init__(self, categorical_columns, numerical_columns):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
    
    def __call__(self, df):
        pass

class SimpleImputation(Imputation):
    def __call__(self, df):
        for c in self.categorical_columns:
            df.loc[df[c].isnull(),[c]] = df[c].dropna().value_counts().index[0]
        
        for c in self.numerical_columns:
            df.loc[df[c].isnull(),[c]] = df[c].dropna().median()

class DatawigImputation(Imputation):
    def __call__(self, df):
        for c in self.categorical_columns + self.numerical_columns:
            input_cols = list(set(df.columns) - set([c]))
            imputer = SimpleImputer(input_columns=input_cols, output_column=c)
            missing = df[c].isnull()
            imputer = imputer.fit(df.loc[~missing, :])
            print(f"Imputing {missing.sum()} missing values in column {c}")
            df_tmp = imputer.predict(df)
            df[c] = df_tmp[c + "_imputed"]