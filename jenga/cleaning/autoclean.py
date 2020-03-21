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
                    ],
                    verbose=False):
        
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.text_columns = text_columns
        self.outlier_removal = [OutlierRemoval] + outlier_removal
        self.imputation = [Imputation] + imputation
        self.verbose = verbose
        self.ppp_model = PipelineWithPPP(pipeline, 
            numerical_columns = self.numerical_columns,
            categorical_columns = self.categorical_columns,
            text_columns = self.text_columns,
            num_repetitions=2,
            perturbation_fractions=[.5,.7,.9]).fit_ppp(train_data, train_labels)

    def _print(self,s):
        if self.verbose:
            print(s)

    def __call__(self, df, test_labels=None):
        '''
        Returns 
        df  cleaned data frame
        ppp_score ppp score witout cleaning
        ppp_scores with cleaning
        '''
        predicted_score_no_cleaning = self.ppp_model.predict_ppp(df)
        self._print(f"PPP score no cleaning {predicted_score_no_cleaning}")
        cleaner_results_ppp = []
        cleaner_results_true = []
        for orm in self.outlier_removal:
            outliers = orm(self.categorical_columns, 
                                self.numerical_columns, 
                                self.text_columns)(df.copy(deep=True))
            for c in self.imputation:
                df_copy = df.copy(deep=True)
                if 'outlier_score' in outliers.columns:
                    self._print(f"Setting {outliers['outlier_score'].sum()} to NaN")
                    df_copy.loc[outliers['outlier_score'], :] = np.nan

                df_cleaned = c(self.categorical_columns, 
                                self.numerical_columns)(df_copy)

                cleaned_score = self.ppp_model.predict_ppp(df_cleaned)
                self._print(f"PPP score with cleaning {c}: {cleaned_score:0.3}")
                cleaner_results_ppp.append(cleaned_score)
                if test_labels is not None:
                    cleaned_true_score = self.ppp_model.pipeline.score(df_cleaned, test_labels)
                    self._print(f"True score with cleaning {c}: {cleaned_true_score:0.3}")
                    cleaner_results_true.append(cleaned_true_score)

        best_cleaning_idx = pd.Series(cleaner_results_ppp).idxmax()
        best_outlier_removal = self.outlier_removal[best_cleaning_idx // len(self.outlier_removal)]
        best_imputation = self.imputation[best_cleaning_idx % len(self.imputation)]
        best_score = cleaner_results_ppp[best_cleaning_idx]
        if best_score > predicted_score_no_cleaning:
            outliers = best_outlier_removal(self.categorical_columns, 
                                self.numerical_columns, 
                                self.text_columns)(df.copy(deep=True))
            if 'outlier_score' in outliers.columns:
                df.loc[outliers['outlier_score'], :] = np.nan

            df = best_imputation(self.categorical_columns, 
                            self.numerical_columns,self.text_columns)(df)

            self._print(f"Best outlier removal {best_outlier_removal.__class__} + best imputation {best_imputation.__class__}: {best_score:0.3}")
        else:
            self._print(f"Cleaning did not improve score")
        
        return df, predicted_score_no_cleaning, cleaner_results_ppp, cleaner_results_true
