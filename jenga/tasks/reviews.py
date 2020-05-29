import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline

from jenga.basis import BinaryClassificationTask


# Predict whether a video game review is considered helpful or not
class VideogameReviewsTask(BinaryClassificationTask):

    def __init__(self, seed):
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

        raw_data = pd.read_csv('data/reviews/videogames.tsv', sep='\t', names=columns, dtype=dtypes)
        # We drop this column, the prediction task is to easy otherwise
        raw_data = raw_data.drop(['total_votes'], axis=1)

        raw_data[['product_title', 'review_headline', 'review_body']] = raw_data[
            ['product_title', 'review_headline', 'review_body']].fillna(value='')
        raw_data['title_and_review_text'] = raw_data.product_title + ' ' + raw_data.review_headline + \
            ' ' + raw_data.review_body

        train_data = self.__extract_data(raw_data, '2015-05-04', '2015-06-14')
        test_data = self.__extract_data(raw_data, '2015-06-15', '2015-06-28')
        train_labels = self.__extract_labels(raw_data, '2015-05-04', '2015-06-14')
        test_labels = self.__extract_labels(raw_data, '2015-06-15', '2015-06-28')

        BinaryClassificationTask.__init__(self,
                                          seed,
                                          train_data,
                                          train_labels,
                                          test_data,
                                          test_labels,
                                          categorical_columns=['marketplace', 'customer_id', 'review_id',
                                                               'product_id', 'product_parent', 'vine',
                                                               'verified_purchase'],
                                          numerical_columns=['star_rating'],
                                          text_columns=['product_title', 'review_headline', 'review_body',
                                                        'title_and_review_text']
                                          )

    @staticmethod
    def __extract_data(raw_data, start_date, end_date):
        data_slice = raw_data[(raw_data.review_date >= start_date) & (raw_data.review_date <= end_date)].copy(deep=True)
        data_slice = data_slice.drop(['helpful_votes'], axis=1)
        return data_slice

    @staticmethod
    def __extract_labels(raw_data, start_date, end_date):
        data_slice = raw_data[(raw_data.review_date >= start_date) & (raw_data.review_date <= end_date)].copy(deep=True)
        labels = np.ravel(label_binarize(data_slice.helpful_votes > 0, [True, False]))

        return labels

    def fit_baseline_model(self, train_data, train_labels):

        numerical_attributes = ['star_rating']
        categorical_attributes = ['vine', 'verified_purchase']

        feature_transformation = ColumnTransformer(transformers=[
            ('numerical_features', StandardScaler(), numerical_attributes),
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), categorical_attributes),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=10000), 'title_and_review_text')
        ], sparse_threshold=1.0)

        param_grid = {
            'learner__loss': ['log'],
            'learner__penalty': ['l2', 'l1'],
            'learner__alpha': [0.0001, 0.001, 0.01, 0.1]
        }

        pipeline = Pipeline([
            ('features', feature_transformation),
            ('learner', SGDClassifier(max_iter=1000))])

        search = GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
        return search.fit(train_data, train_labels)
