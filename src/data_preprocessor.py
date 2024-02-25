# src/data/data_preprocessor.py

import pandas as pd
import numpy as np

class DataPreprocessor:
    @staticmethod
    def adjust_df_for_lag_and_trim(df, lag):
        """
        Prepends NaN rows equal to 'lag' at the beginning of the DataFrame and 
        trims the same number of rows from the end to maintain original length.
        
        Parameters:
        - df: DataFrame to be adjusted.
        - lag: Number of periods to lag the data.
        
        Returns:
        - DataFrame with adjusted data.
        """
        nan_rows = pd.DataFrame(np.nan, index=np.arange(lag), columns=df.columns)
        concatenated_df = pd.concat([nan_rows, df]).reset_index(drop=True)
        trimmed_df = concatenated_df.iloc[:-lag]
        trimmed_df.index = df.index
        return trimmed_df