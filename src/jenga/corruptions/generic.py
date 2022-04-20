import numpy as np

from ..basis import DataCorruption, TabularCorruption


# Inject different kinds of missing values
class MissingValues(TabularCorruption):

    def __init__(self, column, fraction, na_value=np.nan, missingness='MCAR'):
        '''
        Corruptions for structured data
        Input:
        column:    column to perturb, string
        fraction:   fraction of rows to corrupt, float between 0 and 1
        na_value:   value
        missingness:   sampling mechanism for corruptions, string in ['MCAR', 'MAR', 'MNAR']
        '''
        self.column = column
        self.fraction = fraction
        self.sampling = missingness
        self.na_value = na_value

    def transform(self, data):
        corrupted_data = data.copy(deep=True)
        rows = self.sample_rows(corrupted_data)
        corrupted_data.loc[rows, [self.column]] = self.na_value
        return corrupted_data


# Missing Values based on the records' "difficulty" for the model
class MissingValuesBasedOnEntropy(DataCorruption):

    def __init__(
        self,
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

        super().__init__()

    def transform(self, data):
        data = data.copy(deep=True)

        cutoff = int(len(data) * (1 - self.fraction))
        probas = self.model.predict_proba(self.data_to_predict_on)

        if self.most_confident:
            affected = probas.max(axis=1).argsort()[:cutoff]

        else:
            # for samples with the smallest maximum probability the model is most uncertain
            affected = probas.max(axis=1).argsort()[-cutoff:]

        data.loc[data.index[affected], self.column] = self.na_value

        return data


# Swapping a fraction of the values between two columns, mimics input errors in forms
# and programming errors during data preparation
class SwappedValues(TabularCorruption):

    def __init__(self, column, fraction, sampling='CAR', swap_with=None):
        super().__init__(column, fraction, sampling)
        self.swap_with = swap_with

    def transform(self, data):
        if self.swap_with is None:
            columns_to_swap = [c for c in data.columns if c != self.column and data[self.column].dtype == data[c].dtype]
            if len(columns_to_swap) > 0:
                self.swap_with = np.random.choice(columns_to_swap)
            else:
                self.swap_with = ""

        if self.swap_with == "":
            print('SwappedValues only works if at least two columns has same dtype')
            return data

        data = data.copy(deep=True)
        rows = self.sample_rows(data)
        tmp_vals = data.loc[rows, self.swap_with].copy(deep=True)
        data.loc[rows, self.swap_with] = data.loc[rows, self.column]
        data.loc[rows, self.column] = tmp_vals

        return data


class CategoricalShift(TabularCorruption):
    def transform(self, data):
        numeric_cols, _ = self.get_dtype(data)

        if self.column in numeric_cols:
            print('CategoricalShift implemented only for categorical variables')
            return data

        data = data.copy(deep=True)
        rows = self.sample_rows(data)
        histogram = data[self.column].value_counts()
        random_other_val = np.random.permutation(histogram.index)
        data.loc[rows, self.column] = data.loc[rows, self.column].replace(histogram.index, random_other_val)
        return data
