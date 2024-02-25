import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add custom directories to the system path for importing classes
sys.path.extend(['../src'])

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from src.data_preprocessor import DataPreprocessor
from src.portfolio_analysis import PortfolioAnalysis

# Define Meta variables
trx_cost = 0.001
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

# Initialize class instances
data_preprocessor = DataPreprocessor()
portfolio_analysis = PortfolioAnalysis(trx_cost=trx_cost, starting_point=StartingPoint)

# Load the data and process it to align dates etc.
os.chdir("/Users/marcobischoff/Library/Mobile Documents/3L68KQB4HG~com~readdle~CommonDocuments/Documents/Banking & Finance/PMP/GARP-Europe-Industries")
fileName = "data/GARP Data Industries Europe.xlsx"  # Change as needed
dataType = 'Industry'
dataStruct = data_preprocessor.load_and_preprocess_data(fileName)

# storing all the data in appropriate variables
rf_prices = dataStruct['Rf']
benchmark_prices = dataStruct['Benchmark_Prices']
prices = dataStruct[dataType + '_Prices']
eps = dataStruct[dataType + '_EPS']
sps = dataStruct[dataType + '_SPS']
bvps = dataStruct[dataType + '_BVPS']
total_debt = dataStruct[dataType + '_TotalDebt']

# the following variables are accounting data and need to be lagged
eps = data_preprocessor.adjust_df_for_lag_and_trim(eps, lag)
sps = data_preprocessor.adjust_df_for_lag_and_trim(sps, lag)
bvps = data_preprocessor.adjust_df_for_lag_and_trim(bvps, lag)
total_debt = data_preprocessor.adjust_df_for_lag_and_trim(total_debt, lag)

# Calculate monthly returns without introducing forward-looking bias, introduce nans to keep the length the same
benchmark_returns = (benchmark_prices / benchmark_prices.shift(1) - 1)
benchmark_returns.iloc[0] = np.nan 
rf_returns = rf_prices / 12 / 100 # to get it into the same level
xs_benchmark_returns = benchmark_returns.iloc[:, 0].squeeze() - rf_returns.iloc[:, 0].squeeze()
prices_returns = prices / prices.shift(1)  - 1
prices_returns.iloc[0] = np.nan  # Setting the first row to NaN

# Calculate financial ratios
roe = eps.div(bvps)
leverage = total_debt.div(bvps)
earnings2price = eps.div(prices)

# Calculate the growth rates
earnings_growth = eps / eps.shift(GrowthTrend) - 1
sales_growth = sps / sps.shift(GrowthTrend) - 1

# Initialize DataFrames for ranks and universes with NaNs and False, respectively
total_months, n_assets = prices.shape  # Assuming 'prices' is a DataFrame with assets as columns

# Initialize DataFrames for ranks and universes
EarningsGrowthRank = pd.DataFrame(np.nan, index=earnings_growth.index, columns=earnings_growth.columns)
SalesGrowthRank = pd.DataFrame(np.nan, index=sales_growth.index, columns=sales_growth.columns)
combinedGrowthRank = pd.DataFrame(np.nan, index=earnings_growth.index, columns=earnings_growth.columns)
GrowthUniverse = pd.DataFrame(False, index=earnings_growth.index, columns=earnings_growth.columns)

ROERank = pd.DataFrame(np.nan, index=roe.index, columns=roe.columns)
Earnings2PriceRank = pd.DataFrame(np.nan, index=earnings2price.index, columns=earnings2price.columns)
LeverageRank = pd.DataFrame(np.nan, index=leverage.index, columns=leverage.columns)
combinedQVRank = pd.DataFrame(np.nan, index=roe.index, columns=roe.columns)
QVUniverse = pd.DataFrame(False, index=roe.index, columns=roe.columns)

# Apply ranking for each month starting from StartingPoint
for month in range(StartingPoint, total_months):
    # Growth metrics
    EarningsGrowthRank.iloc[month, :] = portfolio_analysis.get_score(earnings_growth.iloc[month, :].values)
    SalesGrowthRank.iloc[month, :] = portfolio_analysis.get_score(sales_growth.iloc[month, :].values)
    combinedGrowthRank.iloc[month, :] = EarningsGrowthRank.iloc[month, :] + SalesGrowthRank.iloc[month, :]
    
    # Filter based on median rank
    medianRank = combinedGrowthRank.iloc[month, :].median()
    GrowthUniverse.iloc[month, :] = combinedGrowthRank.iloc[month, :] < medianRank
    
    # QV metrics
    ROERank.iloc[month, :] = portfolio_analysis.get_score(roe.iloc[month, :].values)
    Earnings2PriceRank.iloc[month, :] = portfolio_analysis.get_score(earnings2price.iloc[month, :].values)
    LeverageRank.iloc[month, :] = portfolio_analysis.get_score(leverage.iloc[month, :].values, long_high_values=False)
    
    combinedQVRank.iloc[month, :] = ROERank.iloc[month, :] + Earnings2PriceRank.iloc[month, :] + LeverageRank.iloc[month, :]
    
    # Filter based on median rank
    medianQVRank = combinedQVRank.iloc[month, :].median()
    QVUniverse.iloc[month, :] = combinedQVRank.iloc[month, :] < medianQVRank

# Initialize GARPUniverse with False values (assuming non-selection by default)
GARPUniverse = pd.DataFrame(False, index=prices_returns.index, columns=prices_returns.columns)

# Iterate through each month
for month in range(StartingPoint - 1, total_months):  # Adjusted for zero-based indexing
    intersection = GrowthUniverse.iloc[month, :] & QVUniverse.iloc[month, :]
    
    if intersection.isna().all() or (intersection == 0).all():
        # Select all assets with available price returns if no intersection
        GARPUniverse.iloc[month, :] = ~prices_returns.iloc[month, :].isna()
    else:
        # Use the intersection for GARP Universe selection
        GARPUniverse.iloc[month, :] = intersection

TotalReturns = {
    'Benchmark': None,
    'GARP': None,
    'GrowthUniverse': None,
    'QVUniverse': None
}

ExcessReturns = {
    'GARP': None,
    'GrowthUniverse': None,
    'QVUniverse': None
}

MonthlyTurnover = {
    'GARP': None,
    'GrowthUniverse': None,
    'QVUniverse': None
}

universes = {
    'GARP': GARPUniverse,
    'GrowthUniverse': GrowthUniverse,
    'QVUniverse': QVUniverse
}

Weights = {
    'GARP': None,
    'GrowthUniverse': None,
    'QVUniverse': None
}

# Example: Preallocating TotalReturns['Benchmark'] as a DataFrame
TotalReturns['Benchmark'] = pd.DataFrame(np.nan, index=prices_returns.index, columns=['Benchmark'])
TotalReturns['Benchmark'].iloc[StartingPoint:, 0] = (1 + benchmark_returns.iloc[StartingPoint:].squeeze()).cumprod()

for key in TotalReturns.keys():
    if key != 'Benchmark':  # Benchmark is already handled
        TotalReturns[key] = pd.DataFrame(np.nan, index=prices_returns.index, columns=[key])
        ExcessReturns[key] = pd.DataFrame(np.nan, index=prices_returns.index, columns=[key])
        MonthlyTurnover[key] = pd.DataFrame(np.nan, index=prices_returns.index, columns=[key])

for key, universe in universes.items():
    # Compute weights, returns, excess returns, and turnover for the given universe
    weights = portfolio_analysis.compute_weights(universe)
    cumulative_returns, excess_returns, turnover = portfolio_analysis.compute_returns_for_universe(
        weights, prices_returns, rf_returns)
    
    Weights[key] = weights

    TotalReturns[key].loc[cumulative_returns.index, key] = cumulative_returns.reindex(TotalReturns[key].index)
    ExcessReturns[key].loc[excess_returns.index, key] = excess_returns.reindex(ExcessReturns[key].index)
    MonthlyTurnover[key].loc[turnover.index, key] = turnover.reindex(MonthlyTurnover[key].index)

# Ensure rf_returns is a Series for easier operations
rf_returns_series = rf_returns.squeeze()

# Assuming StartingPoint is the index from where you want to start considering the data
starting_index = prices_returns.index[StartingPoint]  # This assumes prices_returns's index is date-based and matches with rf_returns and xs_returns

all_stats = {}

xs_benchmark_returns_df = xs_benchmark_returns.loc[starting_index:].to_frame()
rf_returns_series_sliced = rf_returns_series.loc[starting_index:]

# Calculate stats for the benchmark
# For the benchmark, factor_xs_returns can be itself or a relevant benchmark factor returns
statsBenchmark = portfolio_analysis.summarize_performance(xs_benchmark_returns_df, rf_returns_series_sliced, xs_benchmark_returns_df, 12)
all_stats['Benchmark'] = statsBenchmark

# Now, loop through each strategy to compute and collect stats
for strategy in ['GARP', 'GrowthUniverse', 'QVUniverse']:
    # Assuming ExcessReturns[strategy] contains the excess returns for the strategy
    xs_returns = ExcessReturns[strategy].loc[starting_index:].squeeze()  # Ensure this is a Series
    # Call summarize_performance for each strategy
    # Ensure factor_xs_returns and xs_returns are DataFrames, rf_returns_series_sliced is a Series
    stats = portfolio_analysis.summarize_performance(xs_returns.to_frame(), rf_returns_series_sliced, xs_benchmark_returns_df, 12)
    
    # Store the stats in the dictionary
    all_stats[strategy] = stats

# Convert the dictionary of statistics into a DataFrame for easy viewing and analysis
# Each key in `all_stats` becomes a column, and each sub-key becomes a row in the DataFrame
stats_df = pd.DataFrame(all_stats)

# Transpose the DataFrame so strategies are columns and stats are rows
stats_df = stats_df.T

# Convert index to datetime if it's not already, assuming the index is PeriodIndex or similar
if not isinstance(TotalReturns['Benchmark'].index, pd.DatetimeIndex):
    TotalReturns['Benchmark'].index = TotalReturns['Benchmark'].index.to_timestamp()

# Assuming your DataFrames' indices are already in datetime format or converted
starting_date = TotalReturns['Benchmark'].index[StartingPoint]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each strategy
for key, df in TotalReturns.items():
    if df is not None:
        ax.plot(df.loc[starting_date:].index, df.loc[starting_date:][key], label=key)

# Formatting the plot
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Cumulative Returns', fontsize=14)
ax.set_title('Strategy Cumulative Returns Comparison', fontsize=16)
ax.legend(loc='best')
ax.grid(True)

# Rotate date labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()