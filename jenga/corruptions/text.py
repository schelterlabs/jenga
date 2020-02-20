import random


class MissingValues:

    def __init__(self, column, fraction, na_value):
        self.column = column
        self.fraction = fraction
        self.na_value = na_value

    def transform(self, data):
        corrupted_data = data.copy(deep=True)
        num_rows = corrupted_data.shape[0]

        row_indexes = [row for row in range(num_rows)]
        num_rows_to_pick = int(round(self.fraction * num_rows))
        affected_indexes = set(random.sample(row_indexes, num_rows_to_pick))
        row_index_indicators = [row in affected_indexes for row in range(num_rows)]

        corrupted_data.loc[row_index_indicators, [self.column]] = self.na_value
        return corrupted_data


class BrokenCharacters:

    def __init__(self, column, fraction):
        self.column = column
        self.fraction = fraction

    def transform(self, data):
        corrupted_data = data.copy(deep=True)

        replacements = {
            'a': 'á',
            'A': 'Á',
            'e': 'é',
            'E': 'É',
            'o': 'ớ',
            'O': 'Ớ',
            'u': 'ú',
            'U': 'Ú'
        }

        for index, row in corrupted_data.iterrows():
            if random.random() < self.fraction:
                column_value = row[self.column]
                for character, replacement in replacements.items():
                    column_value = column_value.replace(character, replacement)
                corrupted_data.at[index, self.column] = column_value

        return corrupted_data
