import numpy as np
import random
from jenga.basis import DataCorruption


# Inject different kinds of missing values
class MissingValues(DataCorruption):

    def __init__(self, column, fraction, na_value, missingness='MCAR'):
        self.column = column
        self.fraction = fraction
        self.na_value = na_value
        self.missingness = missingness
        DataCorruption.__init__(self)

    def transform(self, data):
        corrupted_data = data.copy(deep=True)

        # Missing Completely At Random
        if self.missingness == 'MCAR':
            missing_indices = np.random.rand(len(data)) < self.fraction
        else:
            n_values_to_discard = int(len(data) * min(self.fraction, 1.0))
            perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
            perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

            # Missing At Random
            if self.missingness == 'MAR':
                depends_on_col = np.random.choice(list(set(data.columns) - {self.column}))
                # pick a random percentile of values in other column
                missing_indices = corrupted_data[depends_on_col].sort_values().iloc[perc_idx].index

            # Missing Not At Random
            else:
                # pick a random percentile of values in this column
                missing_indices = corrupted_data[self.column].sort_values().iloc[perc_idx].index

        corrupted_data.loc[missing_indices, [self.column]] = self.na_value
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

        df.loc[df.index[affected], self.column] = self.na_value

        return df


# Swapping a fraction of the values between two columns, mimics input errors in forms
# and programming errors during data preparation
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
