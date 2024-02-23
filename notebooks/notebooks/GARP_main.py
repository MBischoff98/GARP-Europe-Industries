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

common_date_range = pd.date_range(start=common_start, end=common_end, freq='M')

# Instead of reindexing directly to common_date_range, let's align to monthly periods
for sheet, df in dataStruct.items():
    # Convert DataFrame index to monthly periods for alignment
    df.index = df.index.to_period('M')

    # Create a new index based on the common_date_range also converted to periods
    new_index = common_date_range.to_period('M')

    # Reindex the DataFrame using this new monthly period index
    # Note: This step aligns all DataFrames to the same set of monthly periods
    df_reindexed = df.reindex(new_index)

    # Optional: Custom imputation logic here, as needed

    # Store the updated DataFrame back in your data structure
    dataStruct[sheet] = df_reindexed

rf_prices = dataStruct['Rf']
benchmark_prices = dataStruct['Benchmark_Prices']
prices = dataStruct[dataType + '_Prices']
eps = dataStruct[dataType + '_EPS']
sps = dataStruct[dataType + '_SPS']
bvps = dataStruct[dataType + '_BVPS']
total_debt = dataStruct[dataType + '_TotalDebt']

# Convert to numeric and prepend NaN rows for lag
eps = pd.concat([create_nan_rows_df(eps, lag), eps]).iloc[:-lag]
sps = pd.concat([create_nan_rows_df(sps, lag), sps]).iloc[:-lag]
bvps = pd.concat([create_nan_rows_df(bvps, lag), bvps]).iloc[:-lag]
total_debt = pd.concat([create_nan_rows_df(total_debt, lag), total_debt]).iloc[:-lag]

# Calculate monthly returns without introducing forward-looking bias
benchmark_returns = (benchmark_prices / benchmark_prices.shift(1) - 1)
# Set the first value of the returns series to NaN, as there's no prior month to calculate the first return
benchmark_returns.iloc[0] = np.nan

# Assuming 'rf_prices' contains risk-free returns, adjust as per your data structure
rf_returns = rf_prices / 12 / 100

# Subtract risk-free rate from benchmark returns to get excess returns
xs_benchmark_returns = benchmark_returns.iloc[:, 0].squeeze() - rf_returns.iloc[:, 0].squeeze()

# Calculate returns for each asset
prices_returns = prices / prices.shift(1)  - 1
prices_returns.iloc[0] = np.nan  # Setting the first row to NaN

# Calculate financial ratios
roe = eps.div(bvps)
leverage = total_debt.div(bvps)
earnings2price = eps.div(prices)

# Calculate the growth rates
earnings_growth = eps / eps.shift(GrowthTrend) - 1
sales_growth = sps / sps.shift(GrowthTrend) - 1