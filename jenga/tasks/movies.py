import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Dropout, Activation, Lambda
from keras.models import Model
from keras.regularizers import l2


class PreprocessingDecorator:

    def __init__(self, model):
        self.model = model

    def predict(self, test_data):
        X_test = Preprocessing().prepare(test_data)
        return self.model.predict(X_test)


class Preprocessing:

    def prepare(self, data):
        data_copy = data.copy(deep=True)
        data_copy['encoded_genres'] = self.__encode_genres(data_copy.genres)

        X_train = data_copy[['user', 'movie', 'encoded_genres']].values
        X_train_array = [X_train[:, 0], X_train[:, 1], np.concatenate(X_train[:, 2]).reshape((len(X_train), 18))]

        return X_train_array

    def __encode_genres(self, genre_strs):

        genres = {
            'Action': 0,
            'Adventure': 1,
            'Animation': 2,
            'Children': 3,
            'Comedy': 4,
            'Crime': 5,
            'Documentary': 6,
            'Drama': 7,
            'Fantasy': 8,
            'Film-Noir': 9,
            'Horror': 10,
            'Musical': 11,
            'Mystery': 12,
            'Romance': 13,
            'Sci-Fi': 14,
            'Thriller': 15,
            'War': 16,
            'Western': 17
        }

        all_genre_strs = list(genre_strs)

        encodings = []

        pos = 0
        for genre_str in all_genre_strs:

            one_hot_encoded = np.zeros(18)

            for genre_desc in genre_str.split('|'):
                if genre_desc in genres:
                    one_hot_encoded[genres[genre_desc]] = 1.0

            encodings.append(one_hot_encoded)
            pos += 1

        return encodings


class PredictMovieRatingsTask:

    def __init__(self):
        ratings = pd.read_csv('data/movie-ratings/ratings.csv')
        movies = pd.read_csv('data/movie-ratings/movies.csv')

        ratings_and_genres = ratings.join(movies, on='movieId', how='left', lsuffix='_ratings', rsuffix='_movies')
        ratings_and_genres = ratings_and_genres[['userId', 'movieId_ratings', 'rating', 'timestamp', 'genres']]
        ratings_and_genres.rename(columns={'movieId_ratings': 'movieId'}, inplace=True)
        ratings_and_genres.genres = ratings_and_genres.genres.fillna('')

        ratings_and_genres['date_time'] = pd.to_datetime(ratings_and_genres.timestamp, unit='s')
        ratings_and_genres['year'] = pd.DatetimeIndex(ratings_and_genres['date_time']).year

        ratings_and_genres['user'] = LabelEncoder().fit_transform(ratings_and_genres['userId'].values)
        ratings_and_genres['movie'] = LabelEncoder().fit_transform(ratings_and_genres['movieId'].values)
        ratings_and_genres['rating'] = ratings_and_genres['rating'].values.astype(np.float32)

        ratings_and_genres.drop(['userId', 'movieId', 'timestamp', 'date_time'], axis=1, inplace=True)

        self.num_users = ratings_and_genres['user'].nunique()
        self.num_movies = ratings_and_genres['movie'].nunique()

        self.min_rating = min(ratings_and_genres['rating'])
        self.max_rating = max(ratings_and_genres['rating'])

        self.__interactions = []
        self.__ratings = []

        for year_start, year_end in [(1996, 2016), (2016, 2017), (2017, 2018), (2018, 2019)]:
            interactions, ratings = self.__slice_out_data(ratings_and_genres, year_start, year_end)
            self.__interactions.append(interactions)
            self.__ratings.append(ratings)

        self.__current_slice = -1

    def advance_current_year(self):
        if self.__current_slice < len(self.__interactions) - 2:
            self.__current_slice += 1
            return True
        else:
            return False

    def current_new_train_data(self):
        return self.__interactions[self.__current_slice].copy(deep=True)

    def current_new_train_ratings(self):
        return np.copy(self.__ratings[self.__current_slice])

    def current_test_data(self):
        return self.__interactions[self.__current_slice + 1].copy(deep=True)

    def current_accumulated_train_data(self):

        train_data = None

        for index in range(0, self.__current_slice + 1):
            if train_data is None:
                train_data = self.__interactions[index].copy(deep=True)
            else:
                train_data = train_data.append(self.__interactions[index].copy(deep=True))

        return train_data

    def current_accumulated_train_ratings(self):

        train_ratings = None

        for index in range(0, self.__current_slice + 1):
            if train_ratings is None:
                train_ratings = np.copy(self.__ratings[index])
            else:
                train_ratings = np.concatenate((train_ratings, np.copy(self.__ratings[index])), axis=None)

        return train_ratings

    def __slice_out_data(self, all_data, year_start_inclusive, year_end_exclusive):
        data = all_data[(all_data.year >= year_start_inclusive) & (all_data.year < year_end_exclusive)].copy(deep=True)
        ratings = data.copy(deep=True).rating
        data.drop(['rating'], axis=1, inplace=True)

        return data, ratings

    def fit_baseline_model(self, train_data, train_ratings):
        n_factors = 20

        user_input = Input(shape=(1,))
        user_embedding = Embedding(self.num_users, n_factors,
                                   embeddings_initializer='he_normal',
                                   embeddings_regularizer=l2(1e-6))(user_input)
        user_embedding = Reshape((n_factors,))(user_embedding)

        item_input = Input(shape=(1,))
        item_embedding = Embedding(self.num_movies, n_factors,
                                   embeddings_initializer='he_normal',
                                   embeddings_regularizer=l2(1e-6))(item_input)
        item_embedding = Reshape((n_factors,))(item_embedding)

        genre_input = Input(shape=(18,))

        genres = Dense(10, kernel_initializer='he_normal')(genre_input)
        genres = Activation('relu')(genres)
        genres = Dropout(0.1)(genres)

        x = Concatenate()([user_embedding, item_embedding, genres])
        x = Dropout(0.05)(x)

        x = Dense(10, kernel_initializer='he_normal')(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(1, kernel_initializer='he_normal')(x)
        x = Activation('sigmoid')(x)
        x = Lambda(lambda x: x * (self.max_rating - self.min_rating) + self.min_rating)(x)

        model = Model([user_input, item_input, genre_input], x)
        model.compile('adam', 'mean_squared_error')

        X_train = Preprocessing().prepare(train_data)

        model.fit(X_train, train_ratings, epochs=15, verbose=1)

        return PreprocessingDecorator(model)

    def score_on_test_ratings(self, predicted_ratings):
        return mean_squared_error(self.__ratings[self.__current_slice + 1], predicted_ratings, squared=True)
