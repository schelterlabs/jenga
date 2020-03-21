import random
import numpy as np
import pandas as pd

class SwapValues:

    def __init__(self, fraction, column):
        self.fraction = fraction
        self.column = column

    def __call__(self, clean_df):
        # we operate on a copy of the data
        df = clean_df.copy(deep=True)

        values = df[self.column].value_counts()
        if len(values)>1:
            rows = np.random.uniform(size=len(df))<self.fraction
            values = values.sample(n=2).index.tolist()
            df.loc[rows, self.column] = df.loc[rows, self.column].replace([values[0]],[values[1]])

        return df

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"
