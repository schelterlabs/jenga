import numpy as np
import pandas as pd
import random
from jenga.basis import DataCorruption, TabularCorruption


# Inject different kinds of missing values
class MissingValues(TabularCorruption):

    def transform(self, data):
        corrupted_data = data.copy(deep=True)
        rows = self.sample_rows(corrupted_data)
        corrupted_data.loc[rows, [self.column]] = np.nan
        return corrupted_data


# Missing Values based on the records' "difficulty" for the model
class MissingValuesBasedOnEntropy(DataCorruption):

    def __init__(self,
                 column,
                 fraction,
                 most_confident,
                 model,
                 data_to_predict_on,
                 na_value
                 ):
        self.column = column
        self.fraction = fraction
        self.most_confident = most_confident
        self.model = model
        self.data_to_predict_on = data_to_predict_on
        self.na_value = na_value
        DataCorruption.__init__(self)

    def transform(self, data):
        df = data.copy(deep=True)

        cutoff = int(len(df) * (1 - self.fraction))
        probas = self.model.predict_proba(self.data_to_predict_on)

        if self.most_confident:
            affected = probas.max(axis=1).argsort()[:cutoff]
        else:
            # for samples with the smallest maximum probability the model is most uncertain
            affected = probas.max(axis=1).argsort()[-cutoff:]

        df.loc[df.index[affected], self.column] = np.nan

        return df


# Swapping a fraction of the values between two columns, mimics input errors in forms
# and programming errors during data preparation
class SwappedValues(TabularCorruption):

    def transform(self, data):
        df = data.copy(deep=True)        

        swap_col = np.random.choice([c for c in data.columns if c != self.column])
        
        rows = self.sample_rows(df)

        tmp_vals = df.loc[rows,swap_col].copy(deep=True)
        df.loc[rows, swap_col] = df.loc[rows, self.column]
        df.loc[rows, self.column] = tmp_vals

        return df

class CategoricalShift(TabularCorruption):
    def transform(self, data):
        df = data.copy(deep=True)
        numeric_cols, non_numeric_cols = self.get_dtype(df)
        if self.column in numeric_cols:
            print('CategoricalShift implemented only for categorical variables')
            return df
        else:
            histogram = df[self.column].value_counts()
            random_other_val = np.random.permutation(histogram.index)
            df[self.column] = df[self.column].replace(histogram.index, random_other_val)
            return df



