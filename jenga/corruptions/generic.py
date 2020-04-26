import numpy as np
from jenga.basis import DataCorruption


class MissingValues(DataCorruption):

    def __init__(self, column, fraction, na_value, missingness='MCAR'):
        self.column = column
        self.fraction = fraction
        self.na_value = na_value
        self.missingness = missingness
        DataCorruption.__init__(self)

    def transform(self, data):
        corrupted_data = data.copy(deep=True)

        # TODO check if missingness is valid

        if self.missingness == 'MCAR':
            missing_indices = np.random.rand(len(data)) > self.fraction
        else:
            n_values_to_discard = int(len(data) * min(self.fraction, 1.0))
            perc_lower_start = np.random.randint(0, len(data) - n_values_to_discard)
            perc_idx = range(perc_lower_start, perc_lower_start + n_values_to_discard)

            if self.missingness == 'MAR':
                depends_on_col = np.random.choice(list(set(data.columns) - {self.column}))
                # pick a random percentile of values in other column
                missing_indices = corrupted_data[depends_on_col].sort_values().iloc[perc_idx].index
            else:
                # self.missingness == 'MNAR':
                # pick a random percentile of values in this column
                missing_indices = corrupted_data[self.column].sort_values().iloc[perc_idx].index

        corrupted_data.loc[missing_indices, [self.column]] = self.na_value
        return corrupted_data



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


