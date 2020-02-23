import pandas as pd
from .imputation import SimpleImputation, DatawigImputation
from .ppp import PipelineWithPPP

class AutoClean:
    def __init__(self,  
                    train_data,
                    train_labels,
                    pipeline,
                    numerical_columns=[],
                    categorical_columns=[], 
                    text_columns=[],
                    cleaners=[
                        SimpleImputation,
                        DatawigImputation
                    ]):
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.cleaners = cleaners
        self.ppp_model = PipelineWithPPP(pipeline, 
            numerical_columns = self.numerical_columns,
            categorical_columns = self.categorical_columns,
            text_columns = self.text_columns,
            num_repetitions=3,
            perturbation_fractions=[.1,.5,.9]).fit_ppp(train_data, train_labels)


    def __call__(self, df):
        '''
        Returns 
        df  cleaned data frame
        ppp_score ppp score witout cleaning
        ppp_scores with cleaning
        '''
        predicted_score = self.ppp_model.predict_ppp(df)
        print(f"PPP score no cleaning {predicted_score}")
        cleaner_results = []
        for c in self.cleaners:
            df_cleaned = c(self.categorical_columns, self.numerical_columns)(df.copy(deep=True))
            cleaned_score = self.ppp_model.predict_ppp(df_cleaned)
            print(f"PPP score with cleaning {c.__class__.__name__}: {cleaned_score}")
            cleaner_results.append(cleaned_score)

        best_cleaning_idx = pd.Series(cleaner_results).argmax()
        best_cleaner = self.cleaners[best_cleaning_idx]
        best_score = cleaner_results[best_cleaning_idx]
        if best_score > predicted_score:
            print(f"Best cleaning {best_cleaner.__class__.__name__}: {best_score}")
            df = best_cleaner(self.categorical_columns, self.numerical_columns)(df.copy(deep=True))
        else:
            print(f"Cleaning did not improve score")
        
        return df, predicted_score, cleaner_results
