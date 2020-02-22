import pandas as pd
from .imputation import SimpleImputation, DatawigImputation

class AutoClean:
    def __init__(self,  
                    ppp_model,
                    numerical_columns=[],
                    categorical_columns=[], 
                    text_columns=[]):
        self.ppp_model = ppp_model
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.cleaners = [
            SimpleImputation,
            DatawigImputation
        ]

    def __call__(df):
        predicted_score = ppp_model.predict_ppp(df)
        print(f"PPP score no cleaning {predicted_score}")
        cleaner_results = []
        for c in self.cleaners:
            df_cleaned = c(self.categorical_columns, self.numerical_columns)(df.copy(deep=True))
            cleaned_score = ppp_model.predict_ppp(df_cleaned)
            print(f"PPP score with cleaning {c.__class__}: {cleaned_score}")
            cleaner_results.append(cleaned_score)

        best_cleaning_idx = pd.Series(cleaner_results).argmax()
        best_cleaner = self.cleaners[best_cleaning_idx]
        best_score = cleaner_results[best_cleaning_idx]
        if best_score > predicted_score:
            print(f"Best cleaning {best_cleaner.__class__}: {best_score}")
            return best_cleaner(self.categorical_columns, self.numerical_columns)(df.copy(deep=True))
        else:
            print(f"Cleaning did not improve score")
            return df
