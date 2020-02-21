import random
import numpy as np
import pandas as pd

class SwappedValues:

    def __init__(self, fraction, column_pair):
        self.fraction = fraction
        self.column_pair = column_pair

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        (column_a, column_b) = self.column_pair

        values_of_column_a = list(df[column_a])
        values_of_column_b = list(df[column_b])

        for index in range(0, len(values_of_column_a)):
            if random.random() < self.fraction:
                temp_value = values_of_column_a[index]
                values_of_column_a[index] = values_of_column_b[index]
                values_of_column_b[index] = temp_value

        df[column_a] = values_of_column_a
        df[column_b] = values_of_column_b

        return df

class Outliers:

    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)
        # means = {column: np.mean(df[column]) for column in self.columns}
        stddevs = {column: np.std(df[column]) for column in self.columns}
        scales = {column: random.uniform(1, 5) for column in self.columns}

        if self.fraction > 0:
            for column in self.columns:
                rows = np.random.uniform(size=len(df))<self.fraction
                noise = np.random.normal(0, scales[column] * stddevs[column], size=rows.sum())
                df.loc[rows, column] += noise

        return df


class Scaling:

    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        scale_factor = np.random.choice([10, 100, 1000])
        
        if self.fraction > 0:
            for column in self.columns:
                rows = np.random.uniform(size=len(df))<self.fraction
                df.loc[rows, column] *= scale_factor

        return df