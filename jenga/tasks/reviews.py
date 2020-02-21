import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, label_binarize, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline


class VideogameReviewsTask:

    def __init__(self):
        columns = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent',
                   'product_title', 'product_category', 'star_rating', 'helpful_votes',
                   'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body',
                   'review_date']

        self.numerical_attributes = ['helpful_votes', 'total_votes']
        self.categorical_attributes = ['vine', 'verified_purchase']
        self.text_attributes = 'title_and_review_text'

        raw_data = pd.read_csv('data/reviews/2015-05-videogames.tsv', sep='\t', names=columns)

        raw_data[['product_title', 'review_headline', 'review_body']] = raw_data[
            ['product_title', 'review_headline', 'review_body']].fillna(value='')
        raw_data[
            'title_and_review_text'] = raw_data.product_title + ' ' + raw_data.review_headline + raw_data.review_body

        self.__weeks = [
            self.__extract_data(raw_data, '2015-05-04', '2015-05-10'),
            self.__extract_data(raw_data, '2015-05-11', '2015-05-17'),
            self.__extract_data(raw_data, '2015-05-18', '2015-05-24'),
            self.__extract_data(raw_data, '2015-05-25', '2015-05-31')
        ]

        self.__labels = [
            self.__extract_labels(raw_data, '2015-05-04', '2015-05-10'),
            self.__extract_labels(raw_data, '2015-05-11', '2015-05-17'),
            self.__extract_labels(raw_data, '2015-05-18', '2015-05-24'),
            self.__extract_labels(raw_data, '2015-05-25', '2015-05-31')
        ]

        self.__current_week = -1

    def __extract_data(self, raw_data, start_date, end_date):
        data_slice = raw_data[(raw_data.review_date >= start_date) & (raw_data.review_date <= end_date)].copy(deep=True)
        data_slice = data_slice.drop(['star_rating'], axis=1)
        return data_slice

    def __extract_labels(self, raw_data, start_date, end_date):
        data_slice = raw_data[(raw_data.review_date >= start_date) & (raw_data.review_date <= end_date)].copy(deep=True)
        labels = np.ravel(label_binarize(data_slice.star_rating == 5, [True, False]))

        return labels

    def current_week(self):
        return self.__current_week

    def advance_current_week(self):
        if self.__current_week < len(self.__weeks) - 2:
            self.__current_week += 1
            return True
        else:
            return False

    def current_accumulated_train_data(self):

        train_data = None

        for index in range(0, self.__current_week + 1):
            if train_data is None:
                train_data = self.__weeks[index].copy(deep=True)
            else:
                train_data = train_data.append(self.__weeks[index].copy(deep=True))

        return train_data

    def current_new_train_data(self):
        return self.__weeks[self.__current_week].copy(deep=True)

    def current_new_train_labels(self):
        return np.copy(self.__labels[self.__current_week])

    def current_accumulated_train_labels(self):

        train_labels = None

        for index in range(0, self.__current_week + 1):
            if train_labels is None:
                train_labels = np.copy(self.__labels[index])
            else:
                train_labels = np.concatenate((train_labels, np.copy(self.__labels[index])), axis=None)

        return train_labels

    def current_test_data(self):
        return self.__weeks[self.__current_week + 1].copy(deep=True)

    def score_on_current_test_data(self, predicted_label_probabilities):
        true_labels = self.__labels[self.__current_week + 1]
        return roc_auc_score(true_labels, np.transpose(predicted_label_probabilities)[1])

    def fit_baseline_model(self, train_data, train_labels):

        feature_transformation = ColumnTransformer(transformers=[
            ('numerical_features', StandardScaler(), self.numerical_attributes),
            ('categorical_features', OneHotEncoder(handle_unknown='ignore'), self.categorical_attributes),
            ('textual_features', HashingVectorizer(ngram_range=(1, 3), n_features=10000), self.text_attributes)
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