from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..basis import BinaryClassificationTask


# Predict whether a video game review is considered helpful or not
class VideogameReviewsTask(BinaryClassificationTask):

    def __init__(self, seed):
        """
        Class that represents a binary classification task based on video game reviews.
        Predict whether a video game review is considered helpful or not.

        Args:
            seed (Optional[int], optional): Seed for determinism. Defaults to None.
        """

        columns = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent',
                   'product_title', 'product_category', 'star_rating', 'helpful_votes',
                   'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body',
                   'review_date']
        dtypes = {
            'marketplace': str,
            'customer_id': str,
            'review_id': str,
            'product_id': str,
            'product_parent': str,
            'product_title': str,
            'product_category': str,
            'star_rating': np.int32,
            'helpful_votes': np.int32,
            'total_votes': np.int32,
            'vine': str,
            'verified_purchase': str,
            'review_headline': str,
            'review_body': str,
            'review_date': str
        }
        categorical_columns = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent', 'vine', 'verified_purchase']
        numerical_columns = ['star_rating']
        text_columns = ['product_title', 'review_headline', 'review_body', 'title_and_review_text']

        raw_data = pd.read_csv('../data/reviews/videogames.tsv', sep='\t', names=columns, dtype=dtypes)
        # We drop this column, the prediction task is to easy otherwise
        raw_data = raw_data.drop(['total_votes'], axis=1)

        raw_data[['product_title', 'review_headline', 'review_body']] = raw_data[['product_title', 'review_headline', 'review_body']].fillna(value='')
        raw_data['title_and_review_text'] = raw_data.product_title + ' ' + raw_data.review_headline + ' ' + raw_data.review_body

        train_data = self._extract_data(raw_data, '2015-05-04', '2015-06-14')
        train_labels = self._extract_labels(raw_data, '2015-05-04', '2015-06-14')
        train_labels = train_labels.astype("category")

        test_data = self._extract_data(raw_data, '2015-06-15', '2015-06-28')
        test_labels = self._extract_labels(raw_data, '2015-06-15', '2015-06-28')
        test_labels = test_labels.astype("category")

        super().__init__(
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            text_columns=text_columns,
            is_image_data=False,
            seed=seed
        )

    @staticmethod
    def _extract_data(raw_data, start_date, end_date):
        data_slice = raw_data[(raw_data.review_date >= start_date) & (raw_data.review_date <= end_date)].copy(deep=True)
        data_slice = data_slice.drop(['helpful_votes'], axis=1)
        return data_slice

    @staticmethod
    def _extract_labels(raw_data, start_date, end_date):
        data_slice = raw_data[(raw_data.review_date >= start_date) & (raw_data.review_date <= end_date)].copy(deep=True)
        return (data_slice['helpful_votes'] > 0).replace({True: 1, False: 0})

    def fit_baseline_model(self, train_data: Optional[pd.DataFrame] = None, train_labels: Optional[pd.Series] = None) -> BaseEstimator:
        """
        Because data contains text columns, this overrides the default behavior.

        Fit a baseline model. If no data is given (default), it uses the task's train data and creates the attribute `_baseline_model`. \
            If data is given, it trains this data.

        Args:
            train_data (Optional[pd.DataFrame], optional): Data to train. Defaults to None.
            train_labels (Optional[pd.Series], optional): Labels to train. Defaults to None.

        Raises:
            ValueError: If `train_data` is given but `train_labels` not or vice versa

        Returns:
            BaseEstimator: Trained model
        """

        if (train_data is None and train_labels is not None) or (train_data is not None and train_labels is None):
            raise ValueError("either set both parameters (train_data, train_labels) or non")

        use_original_data = train_data is None

        if use_original_data:
            train_data = self.train_data.copy()
            train_labels = self.train_labels.copy()

        numerical_attributes = ['star_rating']
        categorical_attributes = ['vine', 'verified_purchase']

        feature_transformation = ColumnTransformer(
            transformers=[
                ('numerical_features', StandardScaler(), numerical_attributes),
                ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_attributes),
                ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=10000), 'title_and_review_text')
            ],
            sparse_threshold=1.0
        )

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2'],
            'learner__alpha': [0.00001, 0.0001, 0.001, 0.01]
        }

        pipeline = Pipeline(
            [
                ('features', feature_transformation),
                ('learner', SGDClassifier(max_iter=1000, n_jobs=-1))
            ]
        )

        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
        model = search.fit(train_data, train_labels).best_estimator_

        # only set baseline model attribute if it is trained on the original task data
        if use_original_data:
            self._baseline_model = model

        return model
