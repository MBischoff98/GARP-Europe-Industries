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
    # Check if the DataFrame starts before the common start date and trim
    if df.index.min() < common_start:
        df = df[df.index >= common_start]
    
    # Check if the DataFrame ends after the common end date and trim
    if df.index.max() > common_end:
        df = df[df.index <= common_end]
    
    # Manually set the first index of each DataFrame to match that of rf_prices, if necessary
    # This is useful if there's a specific need to force alignment by index values
    if len(df) > 0:  # Ensure the DataFrame is not empty
        df.index.values[0] = dataStruct['Rf'].index.values[0]
    
    # Update the DataFrame in your data structure
    dataStruct[sheet] = df

rf_prices = dataStruct['Rf']
benchmark_prices = dataStruct['Benchmark_Prices']
prices = dataStruct[dataType + '_Prices']
eps = dataStruct[dataType + '_EPS']
sps = dataStruct[dataType + '_SPS']
bvps = dataStruct[dataType + '_BVPS']
total_debt = dataStruct[dataType + '_TotalDebt']

# Convert to numeric and prepend NaN rows for lag
eps = pd.concat([create_nan_rows_df(eps, lag)])
sps= pd.concat([create_nan_rows_df(sps, lag)])
bvps = pd.concat([create_nan_rows_df(bvps, lag)])
total_debt = pd.concat([create_nan_rows_df(total_debt, lag)])

total_months = len(rf_prices)  # Assuming 'rf_prices' is a pandas DataFrame or Series
n_assets = prices.shape[1]  # Assuming 'prices' is a DataFrame

benchmark_returns = (benchmark_prices.shift(-1) / benchmark_prices - 1)
benchmark_returns.iloc[0] = np.nan  # Adjust the first data point

# Assuming rf_returns are calculated per period (e.g., monthly) from annual data
rf_returns = rf_prices / 12 / 100

# Convert both Series to numpy arrays and then perform subtraction
xs_benchmark_returns = benchmark_returns - rf_returns

prices_returns = prices.shift(-1) / prices - 1
prices_returns.iloc[0] = np.nan  # Setting the first row to NaN aligns the length

# Calculate ROE, Leverage, and Earnings to Price
roe = eps / bvps
leverage = total_debt / bvps
earnings2price = eps / prices

# Create NaN DataFrames for the initial GrowthTrend periods
nan_df_eps = pd.DataFrame(np.nan, index=eps.index[:GrowthTrend], columns=eps.columns)
nan_df_sps = pd.DataFrame(np.nan, index=sps.index[:GrowthTrend], columns=sps.columns)

# Calculate the growth rates as before
earnings_growth = (eps.shift(-GrowthTrend) / eps - 1)
sales_growth = (sps.shift(-GrowthTrend) / sps - 1)

# Concatenate the NaN DataFrames with the calculated growth rates
full_earnings_growth = pd.concat([nan_df_eps, earnings_growth.iloc[GrowthTrend:]])
full_sales_growth = pd.concat([nan_df_sps, sales_growth.iloc[GrowthTrend:]])
