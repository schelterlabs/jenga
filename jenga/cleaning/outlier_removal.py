import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from pyod.models.knn import KNN

class OutlierRemoval:
    def __init__(self, categorical_columns, numerical_columns, text_columns):
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
    
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        transformers = [
            ('numerical_features', numeric_transformer, self.numerical_columns),
            ('categorical_features', categorical_transformer, self.categorical_columns)
        ]

        if self.text_columns:
            transformers.append(('textual_features', 
                                  HashingVectorizer(ngram_range=(1, 3), 
                                                         n_features=2**15), 
                                  ['PyOD_outlier_text_dummy_column']))
            
        self.feature_transformation = ColumnTransformer(transformers=transformers, 
                                                    sparse_threshold=1.0)

    def __call__(self, df):
        return df

     
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"



class PyODKNN(OutlierRemoval):
    def __call__(self, df):

        if self.text_columns:
            df['OutlierRemoval_text_dummy_column'] = ''
            for text_column in self.text_columns:
                df['OutlierRemoval_text_dummy_column'] += " " + df[text_column]

        X = self.feature_transformation.fit_transform(df).to_array()

        clf = KNN()
        df['outlier_score'] = clf.fit_predict(X)
        
        if self.text_columns:
            df = df.drop(['OutlierRemoval_text_dummy_column'], axis=1)
        
        return df

class SKLearnIsolationForest(OutlierRemoval):
    def __call__(self, df):

        if self.text_columns:
            df['OutlierRemoval_text_dummy_column'] = ''
            for text_column in self.text_columns:
                df['OutlierRemoval_text_dummy_column'] += " " + df[text_column]

        X = self.feature_transformation.fit_transform(df)
        clf = IsolationForest(random_state=0, contamination=.1)
        df['outlier_score'] = clf.fit_predict(X) < 0

        if self.text_columns:
            df = df.drop(['OutlierRemoval_text_dummy_column'], axis=1)
        
        return df
   
