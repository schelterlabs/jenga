import pandas as pd
import numpy as np
import random

from .outlier_removal import NoOutlierRemoval, SKLearnIsolationForest
from .imputation import NoImputation, SimpleImputation

class Cleaner:

    def __init__(self, 
                outlier_removal=NoOutlierRemoval,
                imputation=NoImputation,
                verbose=False):
        self.outlier_removal = outlier_removal
        self.imputation = imputation
        self.verbose = verbose

    def __repr__(self):
        return f"{self.__class__.__name__}: OutlierRemoval: {self.outlier_removal.__class__.__name__}, Imputation : {self.imputation.__class__.__name__}"

    def _print(self, s):
        if self.verbose:
            print(s)

    def __call__(self, df):
        df = self.outlier_removal(df)
        if 'outlier_score' in df.columns:
            self._print(f"Setting {df['outlier_score'].sum()} to NaN")
            df.loc[df['outlier_score'], :] = np.nan
            df = df.drop('outlier_score', axis=1)

        return self.imputation(df)