import matplotlib.pyplot as plt
import pandas as pd

class Visualization:
    def __init__(self):
        pass

    @staticmethod
    def plot_cumulative_returns(total_returns, starting_point, title='Strategy Cumulative Returns Comparison'):
        """
        Plots the cumulative returns for given strategies.

        Parameters:
        - total_returns: Dictionary of DataFrames, each representing a strategy's returns over time.
        - starting_point: The point (index) from which to start plotting.
        - title: Title of the plot.
        """
        # Convert index to datetime if needed
        for strategy, returns in total_returns.items():
            if not isinstance(returns.index, pd.DatetimeIndex):
                total_returns[strategy].index = total_returns[strategy].index.to_timestamp()

        # Determine the starting date for plotting
        starting_date = total_returns['Benchmark'].index[starting_point]

        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each strategy
        for key, df in total_returns.items():
            if df is not None:
                ax.plot(df.loc[starting_date:].index, df.loc[starting_date:][key], label=key)

        # Formatting the plot
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Returns', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best')
        ax.grid(True)

        # Rotate date labels for better readability
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_bar_chart(data, title='Bar Chart', xlabel='X-axis', ylabel='Y-axis', legend_title=None):
        """
        Plots a bar chart for given data. The data can be a DataFrame where columns are considered separate categories 
        or a dictionary of Series where each Series represents a category.

        Parameters:
        - data: DataFrame or dictionary of Series with the data to plot.
        - title: Title of the plot.
        - xlabel: Label for the X-axis.
        - ylabel: Label for the Y-axis.
        - legend_title: Title for the legend.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if isinstance(data, dict):  # If data is a dictionary, convert it to a DataFrame for plotting
            data = pd.DataFrame(data)
        
        data.plot(kind='bar', ax=ax, width=0.8)
        
        # Formatting
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=45)
        if legend_title:
            ax.legend(title=legend_title)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
