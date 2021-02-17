import random

from ..basis import DataCorruption


# Mimics cases where text is processed with the wrong encoding
# (e.g., when crawled from the web)
class BrokenCharacters(DataCorruption):

    def __init__(self, column, fraction):
        self.column = column
        self.fraction = fraction
        super().__init__()

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
                    column_value = str(column_value).replace(character, replacement)

                corrupted_data.at[index, self.column] = column_value

        return corrupted_data
