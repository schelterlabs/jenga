import random
import numpy as np

class BrokenCharacters:

    def __init__(self, column, fraction):
        self.column = column
        self.fraction = fraction

    def __call__(self, data):
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
