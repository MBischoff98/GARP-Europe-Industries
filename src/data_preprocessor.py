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
    
    @staticmethod
    def load_and_preprocess_data(fileName):
        """
        Loads data from Excel file, processes each sheet, and aligns them to a common date range.
        
        Parameters:
        - fileName: Path to the Excel file.
        
        Returns:
        - Dictionary of DataFrames with processed data for each sheet.
        """
        xls = pd.ExcelFile(fileName)
        dataStruct = {}
        start_dates = []
        end_dates = []

        for sheet in xls.sheet_names:
            # Read header to get column names
            header_df = pd.read_excel(fileName, sheet_name=sheet, header=3, nrows=0)
            # Read data, skipping initial rows without headers
            df = pd.read_excel(fileName, sheet_name=sheet, skiprows=7, header=None, names=header_df.columns)
            # Set the first column as index
            df.set_index(df.columns[0], inplace=True)
            # Convert index to datetime if necessary
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            dataStruct[sheet] = df
            # Collect start and end dates
            start_dates.append(df.index.min())
            end_dates.append(df.index.max())

        # Determine common date range
        common_start, common_end = max(start_dates), min(end_dates)
        common_date_range = pd.date_range(start=common_start, end=common_end, freq='M')

        # Reindex and align all DataFrames to the common date range
        for sheet, df in dataStruct.items():
            df.index = df.index.to_period('M')  # Align to monthly periods
            new_index = common_date_range.to_period('M')
            dataStruct[sheet] = df.reindex(new_index).to_timestamp()  # Reindex and convert back to timestamp

        return dataStruct