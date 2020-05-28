import random
import numpy as np

from jenga.basis import DataCorruption


# Add gaussian noise to an attribute, mimics noisy, unreliable measurements
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


# Randomly scale a fraction of the values (mimics case where someone actually changes the scale
# of some attribute, e.g., recording a duration in milliseconds instead of seconds)
class Scaling(DataCorruption):

    def __init__(self, column, fraction):
        self.column = column
        self.fraction = fraction
        DataCorruption.__init__(self)

    def transform(self, data):
        df = data.copy(deep=True)

        scale_factor = np.random.choice([10, 100, 1000])

        if self.fraction > 0:
            rows = np.random.uniform(size=len(df)) < self.fraction
            df.loc[rows, self.column] *= scale_factor

        return df

