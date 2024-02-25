import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from statsmodels.api import OLS, add_constant

class PortfolioAnalysis:
    def __init__(self, trx_cost, starting_point):
        self.trx_cost = trx_cost
        self.starting_point = starting_point

    def get_score(self, sort_variable, long_high_values=True):
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

    def compute_weights(self, universe):
        """
        Computes weights for assets in the universe, starting from a specific point.
        
        Parameters:
        - universe: DataFrame indicating asset inclusion in the universe.
        - starting_point: Index from where to start the computation.
        
        Returns:
        - DataFrame of computed weights.
        """

        weights = pd.DataFrame(np.nan, index=universe.index, columns=universe.columns)
        for month in range(self.starting_point, len(universe)):
            number_of_selected_assets = universe.iloc[month, :].sum()
            if number_of_selected_assets > 0:
                weights.iloc[month, :] = universe.iloc[month, :] / number_of_selected_assets
        return weights

    def compute_turnover(self, previous_weights, new_weights, asset_returns, rf):
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

    def compute_returns_for_universe(self, universe_weights, prices_returns, rf_returns):
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
        cumulative_adjusted_returns.iloc[self.starting_point] = 1  # Starting value

        # Adjusted to use the new compute_turnover function
        for month in range(self.starting_point, total_months - 1):
            if month + 1 < total_months:
                previous_weights = universe_weights.iloc[month, :].fillna(0).values
                new_weights = universe_weights.iloc[month + 1, :].fillna(0).values
                asset_returns = prices_returns.iloc[month + 1, :].fillna(0).values
                rf = rf_returns.iloc[month + 1] if month + 1 in rf_returns.index else 0

                # Compute turnover and Rp using the new function
                turnover_val, Rp = self.compute_turnover(previous_weights, new_weights, asset_returns, rf)
                monthly_turnover.iloc[month] = turnover_val
                
                # Calculate strategy returns excluding transaction costs
                monthly_adjusted_returns.iloc[month + 1] = Rp - self.trx_cost * turnover_val
                # Update cumulative returns
                cumulative_adjusted_returns.iloc[month + 1] = cumulative_adjusted_returns.iloc[month] * (1 + monthly_adjusted_returns.iloc[month + 1])
        
        # Calculate excess returns
        xs_returns = monthly_adjusted_returns - rf

        return cumulative_adjusted_returns, xs_returns, monthly_turnover

    def summarize_performance(self, xs_returns, rf, factor_xs_returns, annualization_factor):
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