import random

import pandas as pd
import numpy as np

from ..basis import TabularCorruption


# Add gaussian noise to an attribute, mimics noisy, unreliable measurements
class GaussianNoise(TabularCorruption):
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def transform(self, data):
        _, non_numeric_cols = self.get_dtype(data)

        if self.column in non_numeric_cols or pd.api.types.is_categorical_dtype(data[self.column]):
            print("GaussianNoise implemented only for numerical variables")
            return data

        data = data.copy(deep=True)
        stddev = np.std(data[self.column])
        scale = random.uniform(1, 5)
        rows = self.sample_rows(data)
        noise = np.random.normal(0, scale * stddev, size=len(rows))

        # changing dtype can cause troubles
        original_dtype = data[self.column].dtype
        data.loc[rows, self.column] += noise
        data[self.column] = data[self.column].astype(original_dtype)

        return data


# Randomly scale a fraction of the values (mimics case where someone actually changes the scale
# of some attribute, e.g., recording a duration in milliseconds instead of seconds)
class Scaling(TabularCorruption):
    def transform(self, data):
        _, non_numeric_cols = self.get_dtype(data)

        if self.column in non_numeric_cols or pd.api.types.is_categorical_dtype(data[self.column]):
            print("Scaling implemented only for numerical variables")
            return data

        data = data.copy(deep=True)
        scale_factor = np.random.choice([10, 100, 1000])
        rows = self.sample_rows(data)
        data.loc[rows, self.column] *= scale_factor
        return data
