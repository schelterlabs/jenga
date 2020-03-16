import numpy as np
import pandas as pd
from .imputation import SimpleImputation, DatawigImputation, Imputation
from .outlier_removal import PyODKNN, OutlierRemoval, SKLearnIsolationForest
from .ppp import PipelineWithPPP

class AutoClean:
    def __init__(self,  
                    train_data,
                    train_labels,
                    pipeline,
                    numerical_columns=[],
                    categorical_columns=[], 
                    text_columns=[],
                    outlier_removal=[],
                    imputation=[
                        #SimpleImputation,
                        #DatawigImputation
                    ]):
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.outlier_removal = outlier_removal + [OutlierRemoval]
        self.imputation = imputation + [Imputation]
        self.ppp_model = PipelineWithPPP(pipeline, 
            numerical_columns = self.numerical_columns,
            categorical_columns = self.categorical_columns,
            text_columns = self.text_columns,
            num_repetitions=2,
            perturbation_fractions=[.5,.7,.9]).fit_ppp(train_data, train_labels)


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
        for orm in self.outlier_removal:
            outliers = orm(self.categorical_columns, 
                                self.numerical_columns, 
                                self.text_columns)(df.copy(deep=True))
            for c in self.imputation:
                df_copy = df.copy(deep=True)
                if 'outlier_score' in outliers.columns:
                    df_copy.loc[outliers['outlier_score'], :] = np.nan

                df_cleaned = c(self.categorical_columns, 
                                self.numerical_columns)(df_copy)

                cleaned_score = self.ppp_model.predict_ppp(df_cleaned)
                print(f"PPP score with cleaning {c}: {cleaned_score}")
                cleaner_results.append(cleaned_score)

        best_cleaning_idx = pd.Series(cleaner_results).argmax()
        best_outlier_removal = self.outlier_removal[best_cleaning_idx // len(self.outlier_removal)]
        best_imputation = self.imputation[best_cleaning_idx % len(self.imputation)]
        best_score = cleaner_results[best_cleaning_idx]
        if best_score > predicted_score:
            outliers = best_outlier_removal(self.categorical_columns, 
                                self.numerical_columns, 
                                self.text_columns)(df.copy(deep=True))
            if 'outlier_score' in outliers.columns:
                df.loc[outliers['outlier_score'], :] = np.nan

            df = best_imputation(self.categorical_columns, 
                            self.numerical_columns,self.text_columns)(df)

            print(f"Best cleaning {best_outlier_removal.__class__.__name__} + {best_imputation.__class__.__name__}: {best_score}")
        else:
            print(f"Cleaning did not improve score")
        
        return df, predicted_score, cleaner_results
