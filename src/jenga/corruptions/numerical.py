import random

import numpy as np

from ..basis import TabularCorruption


# Add gaussian noise to an attribute, mimics noisy, unreliable measurements
class GaussianNoise(TabularCorruption):
    def __str__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"

    def transform(self, data):
        _, non_numeric_cols = self.get_dtype(data)

        if self.column in non_numeric_cols:
            print("GaussianNoise implemented only for numerical variables")
            return data

        data = data.copy(deep=True)
        stddev = np.std(data[self.column])
        scale = random.uniform(1, 5)
        rows = self.sample_rows(data)
        noise = np.random.normal(0, scale * stddev, size=len(rows))
        data.loc[rows, self.column] += noise
        return data


# Randomly scale a fraction of the values (mimics case where someone actually changes the scale
# of some attribute, e.g., recording a duration in milliseconds instead of seconds)
class Scaling(TabularCorruption):
    def transform(self, data):
        _, non_numeric_cols = self.get_dtype(data)

        if self.column in non_numeric_cols:
            print("Scaling implemented only for numerical variables")
            return data

        data = data.copy(deep=True)
        scale_factor = np.random.choice([10, 100, 1000])
        rows = self.sample_rows(data)
        data.loc[rows, self.column] *= scale_factor
        return data
