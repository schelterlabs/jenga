import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from jenga.basis import BinaryClassificationTask

# some binary classification tasks from
# https://www.openml.org/search?q=qualities.NumberOfClasses%3A2%2520qualities.NumberOfInstances%3Alt%3B10000%2520qualities.NumberOfFeatures%3Alt%3B100&type=data
OPENML_IDS = [1448, 40994]


# Predict whether a person has high or low income based on demographic and financial attributes
class OpenMLTask(BinaryClassificationTask):

    def __init__(self, seed=0, openml_id=1448):
        self.openml_id = openml_id
        X, y = fetch_openml(data_id=self.openml_id, as_frame=True, return_X_y=True)

        self.__contains_missing_values = X.isnull().values.any()  # TODO: does this work in all cases?

        categorical_columns, numeric_columns, text_columns = self._guess_dtypes(X)
        print(f'Found {len(categorical_columns)} categorical columns: {categorical_columns}')
        print(f'Found {len(numeric_columns)} numeric columns: {numeric_columns}')

        train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2)

        super().__init__(
            seed,
            train_data,
            train_labels,
            test_data,
            test_labels,
            categorical_columns=categorical_columns,
            numerical_columns=numeric_columns,
            text_columns=text_columns
        )

    def _is_categorical(self, col, max_unique_ratio=0.05):
        # return len(col.value_counts()) / len(col) < max_unique_ratio
        return pd.api.types.is_categorical_dtype(col)

    def _guess_dtypes(self, df):
        categorical_columns = [c for c in df.columns if self._is_categorical(df[c])]
        numeric_columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in categorical_columns]
        text_columns = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) and c not in categorical_columns and c not in numeric_columns]
        return categorical_columns, numeric_columns, text_columns

    def contains_missing_values(self) -> bool:
        return self.__contains_missing_values

    def fit_baseline_model(self, train_data, train_labels):

        for col in self.categorical_columns:
            train_data[col] = train_data[col].astype(str)

        categorical_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='constant', fill_value='__NA__')),
                ('one_hot_encode', OneHotEncoder(handle_unknown='ignore'))
            ]
        )

        numeric_preprocessing = Pipeline(
            [
                ('mark_missing', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaling',  StandardScaler())
            ]
        )

        feature_transformation = ColumnTransformer(transformers=[
                ('categorical_features', categorical_preprocessing, self.categorical_columns),
                ('scaled_numeric', numeric_preprocessing, self.numerical_columns)
            ]
        )

        param_grid = {
            'learner__loss': ['log'],
            'learner__alpha': [0.0001, 0.001, 0.01]
        }

        pipeline = Pipeline(
            [
                ('features', feature_transformation),
                ('learner', SGDClassifier(max_iter=1000))
            ]
        )

        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, verbose=1, n_jobs=-1)
        model = search.fit(train_data, train_labels)

        return model
