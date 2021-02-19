import pandas as pd
from sklearn.model_selection import train_test_split

from ..basis import BinaryClassificationTask


# Predict whether a person has high or low income based on demographic and financial attributes
class IncomeEstimationTask(BinaryClassificationTask):

    def __init__(self, seed, ignore_incomplete_records_for_training=False):
        """
        Class that represents a binary classification task based on the [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult).

        Adult Data Set Abstract:
            Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

        Args:
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
            ignore_incomplete_records_for_training (bool, optional): Indicates whether or not to drop rows with missing values for training. \
                Defaults to False.
        """

        columns = ['workclass', 'occupation', 'marital_status', 'education', 'hours_per_week', 'age']
        categorical_columns = ['workclass', 'occupation', 'marital_status', 'education']
        numerical_columns = ['hours_per_week', 'age']

        all_data = pd.read_csv('../data/income/adult.csv', na_values='?')

        train_split, test_split = train_test_split(all_data, test_size=0.2)

        if ignore_incomplete_records_for_training:
            train_split = train_split.dropna()

        train_data = train_split[columns]
        train_labels = (train_split['class'] == '>50K').replace({True: 1, False: 0})
        train_labels = train_labels.astype("category")

        test_data = test_split[columns]
        test_labels = (test_split['class'] == '>50K').replace({True: 1, False: 0})
        test_labels = test_labels.astype("category")

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            is_image_data=False,
            seed=seed
        )
