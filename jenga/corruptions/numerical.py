import random
import numpy as np

from jenga.basis import DataCorruption


class SwappedValues(DataCorruption):

    def __init__(self, column_a, column_b, fraction):
        self.column_a = column_a
        self.column_b = column_b
        self.fraction = fraction
        DataCorruption.__init__(self)

    def transform(self, clean_df):
        df = clean_df.copy(deep=True)

        values_of_column_a = list(df[self.column_a])
        values_of_column_b = list(df[self.column_b])

        for index in range(0, len(values_of_column_a)):
            if random.random() < self.fraction:
                temp_value = values_of_column_a[index]
                values_of_column_a[index] = values_of_column_b[index]
                values_of_column_b[index] = temp_value

        df[self.column_a] = values_of_column_a
        df[self.column_b] = values_of_column_b

        return df


class GaussianNoise(DataCorruption):

    def __init__(self, column, fraction):
        self.column = column
        self.fraction = fraction
        DataCorruption.__init__(self)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def transform(self, data):
        df = data.copy(deep=True)
        stddev = np.std(df[self.column])
        scale = random.uniform(1, 5)

        if self.fraction > 0:
            rows = np.random.uniform(size=len(df)) < self.fraction
            noise = np.random.normal(0, scale * stddev, size=rows.sum())
            df.loc[rows, self.column] += noise

        return df


class Scaling(DataCorruption):

    def __init__(self, fraction, columns):
        self.fraction = fraction
        self.columns = columns
        DataCorruption.__init__(self)

    def transform(self, data):
        df = data.copy(deep=True)

        scale_factor = np.random.choice([10, 100, 1000])

        if self.fraction > 0:
            for column in self.columns:
                rows = np.random.uniform(size=len(df)) < self.fraction
                df.loc[rows, column] *= scale_factor

        return df

