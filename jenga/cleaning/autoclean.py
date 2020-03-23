import numpy as np
import pandas as pd
from .imputation import SimpleImputation, DatawigImputation, NoImputation
from .outlier_removal import PyODKNN, NoOutlierRemoval, SKLearnIsolationForest
from .ppp import PipelineWithPPP
from .cleaner import Cleaner

DEFAULT_CLEANER_CANDIDATES = [
    (NoOutlierRemoval, NoImputation),
    (SKLearnIsolationForest, NoImputation),
    (NoOutlierRemoval, SimpleImputation),
    (SKLearnIsolationForest, SimpleImputation)
]

class AutoClean:
    def __init__(self,  
                    train_data,
                    train_labels,
                    pipeline,
                    numeric_columns=[],
                    categorical_columns=[], 
                    text_columns=[],
                    cleaners=DEFAULT_CLEANER_CANDIDATES,
                    verbose=False):
        
        datatypes = {
                    'numeric_columns': numeric_columns,
                    'categorical_columns': categorical_columns,
                    'text_columns': text_columns
                }
        self.verbose = verbose

        self.cleaners = [Cleaner(outlier_removal=orm(**datatypes), imputation=imp(**datatypes)) \
                            for orm, imp in cleaners]
        
        self.ppp_model = PipelineWithPPP(pipeline, verbose=self.verbose, **datatypes).fit_ppp(train_data, train_labels)

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
        for cleaner in self.cleaners:
            df_cleaned = cleaner(df.copy(deep=True))
            cleaned_score = self.ppp_model.predict_ppp(df_cleaned)
            self._print(f"PPP score with cleaning {cleaner}: {cleaned_score:0.3}")
            cleaner_results_ppp.append(cleaned_score)
            if test_labels is not None:
                cleaned_true_score = self.ppp_model.pipeline.score(df_cleaned, test_labels)
                self._print(f"True score with cleaning {cleaner}: {cleaned_true_score:0.3}")
                cleaner_results_true.append(cleaned_true_score)

        best_cleaning_idx = pd.Series(cleaner_results_ppp).idxmax()
        best_score = cleaner_results_ppp[best_cleaning_idx]
        if best_score > predicted_score_no_cleaning:
            df = self.cleaners[best_cleaning_idx](df.copy(deep=True))
            self._print(f"Best outlier removal {self.cleaners[best_cleaning_idx]}: {best_score:0.3}")
        else:
            self._print(f"Cleaning did not improve score")
        
        return df, predicted_score_no_cleaning, cleaner_results_ppp, cleaner_results_true
