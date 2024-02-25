import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from scipy.stats import skew, kurtosis
from statsmodels.api import OLS, add_constant
import sys

# Add custom directories to the system path for importing classes
sys.path.extend(['../src/data', '../src/models', '../src/visualization'])

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)

from src.data.data_preprocessor import DataPreprocessor

def get_score(sort_variable, long_high_values=True):
    """
    Assigns ranks to values, handling NaNs. Higher values get higher ranks if
    long_high_values is True, otherwise lower values get higher ranks.
    
    Parameters:
    - sort_variable: Array-like object of values to rank.
    - long_high_values: Boolean indicating the ranking direction.
    
    Returns:
    - Array of ranks.
    """

    # Initialize scores with NaNs
    scores = np.nan * np.ones_like(sort_variable)
    
    # Find indices of non-NaN elements
    non_nan_indices = np.where(~np.isnan(sort_variable))[0]
    valid_values = sort_variable[non_nan_indices]
    
    # Sort non-NaN elements and get sorted indices
    if long_high_values:
        sorted_indices = np.argsort(-valid_values)
    else:
        sorted_indices = np.argsort(valid_values)
    
    # Assign ranks (1-based indexing) based on sorted order, correctly mapping back to original positions
    ranks = np.arange(1, len(valid_values) + 1)
    scores[non_nan_indices[sorted_indices]] = ranks
    
    return scores

def compute_weights(universe, starting_point):
    """
    Computes weights for assets in the universe, starting from a specific point.
    
    Parameters:
    - universe: DataFrame indicating asset inclusion in the universe.
    - starting_point: Index from where to start the computation.
    
    Returns:
    - DataFrame of computed weights.
    """

    weights = pd.DataFrame(np.nan, index=universe.index, columns=universe.columns)
    for month in range(starting_point, len(universe)):
        number_of_selected_assets = universe.iloc[month, :].sum()
        if number_of_selected_assets > 0:
            weights.iloc[month, :] = universe.iloc[month, :] / number_of_selected_assets
    return weights

def compute_turnover(previous_weights, new_weights, asset_returns, rf):
    """
    Computes turnover and portfolio return excluding transaction costs.
    
    Parameters:
    - previous_weights: np.array; weights of assets in the portfolio at the previous time step.
    - new_weights: np.array; target weights of assets in the portfolio for the current time step.
    - asset_returns: np.array; returns of assets for the current time step.
    - rf: float; risk-free rate for the current time step.
    
    Returns:
    - turnover: float; sum of the absolute differences between new and current weights.
    - Rp: float; portfolio return excluding transaction costs.
    """
    
    # Compute the portfolio return excluding transaction costs
    Rp = np.sum(previous_weights * asset_returns) + (1 - np.sum(previous_weights)) * rf
    
    # Adjust previous weights for asset returns to get current value per asset
    value_per_asset = previous_weights * (1 + asset_returns)
    
    # Calculate current weights based on adjusted asset values
    current_weights = value_per_asset / (1 + Rp)
    
    # Compute turnover as the sum of absolute differences between new and current weights
    turnover = np.sum(np.abs(new_weights - current_weights))

    return turnover, Rp

def compute_returns_for_universe(universe_weights, prices_returns, rf_returns, trx_cost, starting_point):
    """
    Computes cumulative returns, excess returns, and turnover for a given universe.
    
    Parameters:
    - universe_weights: DataFrame of asset weights over time.
    - prices_returns: DataFrame of asset returns.
    - rf_returns: Series of risk-free rates.
    - trx_cost: Transaction cost rate.
    - starting_point: Index from where to start the computation.
    
    Returns:
    - Tuple of (cumulative returns, excess returns, monthly turnover).
    """

    total_months, _ = prices_returns.shape
    cumulative_adjusted_returns = pd.Series(np.nan, index=prices_returns.index)
    monthly_turnover = pd.Series(np.nan, index=prices_returns.index)
    monthly_adjusted_returns = pd.Series(np.nan, index=prices_returns.index)

    # Initialize cumulative adjusted returns
    cumulative_adjusted_returns.iloc[starting_point] = 1  # Starting value

    # Adjusted to use the new compute_turnover function
    for month in range(starting_point, total_months - 1):
        if month + 1 < total_months:
            previous_weights = universe_weights.iloc[month, :].fillna(0).values
            new_weights = universe_weights.iloc[month + 1, :].fillna(0).values
            asset_returns = prices_returns.iloc[month + 1, :].fillna(0).values
            rf = rf_returns.iloc[month + 1] if month + 1 in rf_returns.index else 0

            # Compute turnover and Rp using the new function
            turnover_val, Rp = compute_turnover(previous_weights, new_weights, asset_returns, rf)
            monthly_turnover.iloc[month] = turnover_val
            
            # Calculate strategy returns excluding transaction costs
            monthly_adjusted_returns.iloc[month + 1] = Rp - trx_cost * turnover_val
            # Update cumulative returns
            cumulative_adjusted_returns.iloc[month + 1] = cumulative_adjusted_returns.iloc[month] * (1 + monthly_adjusted_returns.iloc[month + 1])
    
    # Calculate excess returns
    xs_returns = monthly_adjusted_returns - rf

    return cumulative_adjusted_returns, xs_returns, monthly_turnover

def summarize_performance(xs_returns, rf, factor_xs_returns, annualization_factor):
    """
    Computes various performance statistics for investment strategies.
    
    Parameters:
    - xs_returns: DataFrame of excess returns.
    - rf: Series of risk-free rates.
    - factor_xs_returns: DataFrame of factor excess returns.
    - annualization_factor: Factor to annualize the statistics.
    
    Returns:
    - Dictionary of computed statistics.
    """

    # Ensure inputs are DataFrame for consistent processing
    if isinstance(xs_returns, pd.Series):
        xs_returns = xs_returns.to_frame()
    if isinstance(factor_xs_returns, pd.Series):
        factor_xs_returns = factor_xs_returns.to_frame()
    
    # Compute total returns
    total_returns = xs_returns + rf.values.reshape(-1, 1)

    # Compute the terminal value of the portfolios to get the geometric mean return per period
    final_pf_val_rf = (1 + rf).prod()
    final_pf_val_total_ret = (1 + total_returns).prod()
    geom_avg_rf = 100 * (final_pf_val_rf ** (annualization_factor / len(rf)) - 1)
    geom_avg_total_return = 100 * (final_pf_val_total_ret ** (annualization_factor / total_returns.shape[0]) - 1)
    geom_avg_xs_return = geom_avg_total_return - geom_avg_rf

    # Regress returns on benchmark to get alpha and factor exposures
    X = add_constant(factor_xs_returns)
    betas = {}
    alpha_geometric = {}
    for column in xs_returns:
        model = OLS(xs_returns[column], X).fit()
        betas[column] = model.params[1:]
        # Compute the total return on the passive alternative and the annualized alpha
        bm_ret = factor_xs_returns.dot(model.params[1:]) + rf
        final_pf_val_bm = (1 + bm_ret).prod()
        geom_avg_bm_return = 100 * (final_pf_val_bm ** (annualization_factor / len(bm_ret)) - 1)
        alpha_geometric[column] = geom_avg_total_return[column] - geom_avg_bm_return

    # Rescale the returns to be in percentage points
    xs_returns *= 100
    total_returns *= 100

    # Compute first three autocorrelations
    autocorrelations = xs_returns.apply(lambda x: [x.autocorr(lag) for lag in range(1, 4)])

    # Calculate the statistics
    stats = {
        'ArithmAvgTotalReturn': annualization_factor * total_returns.mean(),
        'ArithmAvgXsReturn': annualization_factor * xs_returns.mean(),
        'StdXsReturns': np.sqrt(annualization_factor) * xs_returns.std(),
        'SharpeArithmetic': (annualization_factor * xs_returns.mean()) / (np.sqrt(annualization_factor) * xs_returns.std()),
        'GeomAvgTotalReturn': geom_avg_total_return,
        'GeomAvgXsReturn': geom_avg_xs_return,
        'SharpeGeometric': geom_avg_xs_return / (np.sqrt(annualization_factor) * xs_returns.std()),
        'MinXsReturn': xs_returns.min(),
        'MaxXsReturn': xs_returns.max(),
        'SkewXsReturn': xs_returns.apply(skew),
        'KurtXsReturn': xs_returns.apply(kurtosis),
        'AlphaArithmetic': annualization_factor * 100 * pd.DataFrame(betas).loc[0].values,
        'AlphaGeometric': pd.Series(alpha_geometric),
        'Betas': pd.DataFrame(betas),
        'Autocorrelations': pd.DataFrame(autocorrelations).T
    }

    return stats

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

for sheet in xls.sheet_names:
    # Read just the header row first to get column names (row 4 in Excel, index 3 for pandas)
    header_df = pd.read_excel(fileName, sheet_name=sheet, header=3, nrows=0)
    
    # Now, read the full sheet, skipping the first 7 rows, without headers
    df = pd.read_excel(fileName, sheet_name=sheet, skiprows=7)
    
    # Assign the column names from the header row
    df.columns = header_df.columns
    
    # If the first column is dates and should be the index
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
eps = adjust_df_for_lag_and_trim(eps, lag)
sps = adjust_df_for_lag_and_trim(sps, lag)
bvps = adjust_df_for_lag_and_trim(bvps, lag)
total_debt = adjust_df_for_lag_and_trim(total_debt, lag)

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

# Initialize DataFrames for ranks and universes with NaNs and False, respectively
total_months, n_assets = prices.shape  # Assuming 'prices' is a DataFrame with assets as columns
starting_point = 90  # Assuming you start ranking and filtering from this month

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
    EarningsGrowthRank.iloc[month, :] = get_score(earnings_growth.iloc[month, :].values)
    SalesGrowthRank.iloc[month, :] = get_score(sales_growth.iloc[month, :].values)
    combinedGrowthRank.iloc[month, :] = EarningsGrowthRank.iloc[month, :] + SalesGrowthRank.iloc[month, :]
    
    # Filter based on median rank
    medianRank = combinedGrowthRank.iloc[month, :].median()
    GrowthUniverse.iloc[month, :] = combinedGrowthRank.iloc[month, :] < medianRank
    
    # QV metrics
    ROERank.iloc[month, :] = get_score(roe.iloc[month, :].values)
    Earnings2PriceRank.iloc[month, :] = get_score(earnings2price.iloc[month, :].values)
    LeverageRank.iloc[month, :] = get_score(leverage.iloc[month, :].values, long_high_values=False)
    
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
    'GARP_high_trx_costs': None,
    'GrowthUniverse': None,
    'QVUniverse': None
}

ExcessReturns = {
    'GARP': None,
    'GARP_high_trx_costs': None,
    'GrowthUniverse': None,
    'QVUniverse': None
}

MonthlyTurnover = {
    'GARP': None,
    'GARP_high_trx_costs': None,
    'GrowthUniverse': None,
    'QVUniverse': None
}
# Example: Preallocating TotalReturns['Benchmark'] as a DataFrame
TotalReturns['Benchmark'] = pd.DataFrame(np.nan, index=prices_returns.index, columns=['Benchmark'])
# Updating 'Benchmark' after preallocation as a DataFrame
TotalReturns['Benchmark'].iloc[StartingPoint:, 0] = (1 + benchmark_returns.iloc[StartingPoint:].squeeze()).cumprod()

# Preallocate DataFrames with NaNs for TotalReturns, ExcessReturns, and MonthlyTurnover
for key in TotalReturns.keys():
    if key != 'Benchmark':  # Benchmark is already handled
        TotalReturns[key] = pd.DataFrame(np.nan, index=prices_returns.index, columns=[key])
        ExcessReturns[key] = pd.DataFrame(np.nan, index=prices_returns.index, columns=[key])
        MonthlyTurnover[key] = pd.DataFrame(np.nan, index=prices_returns.index, columns=[key])

universes = {
    'GARP': GARPUniverse,
    'GARP_high_trx_costs': GARPUniverse,  # Assuming same universe for high trx costs example
    'GrowthUniverse': GrowthUniverse,
    'QVUniverse': QVUniverse
}

transaction_costs = {
    'GARP': trx_cost,
    'GARP_high_trx_costs': trx_costs_high,
    'GrowthUniverse': trx_cost,
    'QVUniverse': trx_cost
}

Weights = {
    'GARP': None,
    'GARP_high_trx_costs': None,
    'GrowthUniverse': None,
    'QVUniverse': None
}

for key, universe in universes.items():
    # Compute weights, returns, excess returns, and turnover for the given universe
    weights = compute_weights(universe, StartingPoint)
    cumulative_returns, excess_returns, turnover = compute_returns_for_universe(
        weights, prices_returns, rf_returns, transaction_costs[key], StartingPoint)
    
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
statsBenchmark = summarize_performance(xs_benchmark_returns_df, rf_returns_series_sliced, xs_benchmark_returns_df, 12)
all_stats['Benchmark'] = statsBenchmark

# Now, loop through each strategy to compute and collect stats
for strategy in ['GARP', 'GARP_high_trx_costs', 'GrowthUniverse', 'QVUniverse']:
    # Assuming ExcessReturns[strategy] contains the excess returns for the strategy
    xs_returns = ExcessReturns[strategy].loc[starting_index:].squeeze()  # Ensure this is a Series
    # Call summarize_performance for each strategy
    # Ensure factor_xs_returns and xs_returns are DataFrames, rf_returns_series_sliced is a Series
    stats = summarize_performance(xs_returns.to_frame(), rf_returns_series_sliced, xs_benchmark_returns_df, 12)
    
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