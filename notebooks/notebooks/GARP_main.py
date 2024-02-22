import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Create NaN rows DataFrame
def create_nan_rows_df(df, lag):
    nan_rows = pd.DataFrame(np.nan, index=range(lag), columns=df.columns[1:])  # Skip first column if needed
    return nan_rows
    

# Define Meta variables
trx_cost = 0.001
trx_costs_high = 0.0025
GrowthTrend = 36
lag = 4
StartingPoint = 90

# Define color palette
colors = {
    'GARP': 'navy',  # Dark Blue
    'Growth': 'darkred',  # Wine Red
    'QV': '#54ff9f',  # Neon Green
    'Benchmark': 'darkgrey'  # Dark Grey
}

os.chdir("/Users/marcobischoff/Library/Mobile Documents/3L68KQB4HG~com~readdle~CommonDocuments/Documents/Banking & Finance/PMP/GARP-Europe-Industries")
fileName = "data/GARP Data Industries Europe.xlsx"  # Change as needed
if 'Sectors' in fileName:
    dataType = 'Sector'
elif 'Industries' in fileName:
    dataType = 'Industry'
else:
    raise ValueError('Invalid file name. The file should be either Sectors or Industries data.')

xls = pd.ExcelFile(fileName)
sheets = xls.sheet_names
dataStruct = {}

for sheet in sheets:
    # Skip the first 7 rows and parse the first column as dates
    df = pd.read_excel(fileName, sheet_name=sheet, skiprows=7, parse_dates=[0])
    # Set the first column (now dates) as the index
    df.set_index(df.columns[0], inplace=True)
    # Store in your dictionary
    dataStruct[sheet] = df

start_dates = []
end_dates = []

for df in dataStruct.values():
    start_dates.append(df.index.min())
    end_dates.append(df.index.max())

common_start = max(start_dates)
common_end = min(end_dates)

for sheet, df in dataStruct.items():
    # Create a new date range that covers the common period
    common_dates = pd.date_range(start=common_start, end=common_end, freq='M')  # 'D' for daily frequency
    # Reindex the DataFrame to the new date range
    dataStruct[sheet] = df.reindex(common_dates)

for sheet, df in dataStruct.items():
    # Create a new date range that covers the common period
    common_dates = pd.date_range(start=common_start, end=common_end, freq='M')  # 'D' for daily frequency
    # Reindex the DataFrame to the new date range
    dataStruct[sheet] = df.reindex(common_dates)

df.fillna(method='ffill', inplace=True)

for sheet, df in dataStruct.items():
    print(f"{sheet}: Start = {df.index.min()}, End = {df.index.max()}")

rf_prices = dataStruct['Rf']
benchmark_prices = dataStruct['Benchmark_Prices']
prices = dataStruct[dataType + '_Prices']
eps = dataStruct[dataType + '_EPS']
sps = dataStruct[dataType + '_SPS']
bvps = dataStruct[dataType + '_BVPS']
total_debt = dataStruct[dataType + '_TotalDebt']

# Convert to numeric and prepend NaN rows for lag
eps_numeric = pd.concat([create_nan_rows_df(eps, lag)])
sps_numeric = pd.concat([create_nan_rows_df(sps, lag)])
bvps_numeric = pd.concat([create_nan_rows_df(bvps, lag)])
total_debt_numeric = pd.concat([create_nan_rows_df(total_debt, lag)])
