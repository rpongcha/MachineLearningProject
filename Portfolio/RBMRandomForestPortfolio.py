from BasePortfolio import BasePortfolio
import pandas as pd
import math


class RBMRandomForestPortfolio(BasePortfolio):

    def __init__(self, symbol, bars, signals, initial_capital=100000.0, minimum_cash_on_hand=0.0):
        """
        Construct of the class
        :param symbol:
        :param bars:
        :param signals:
        :param initial_capital:
        :param minimum_cash_on_hand:
        :return:
        """
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.minimum_cash = float(minimum_cash_on_hand)
        # the purchasing size is fixed at 100 shares
        self.purchasing_size = 100.
        # generate positions from the input signals
        self.positions = self.generate_positions()

    def generate_positions(self):
        """
        generate positions from the signals
        :return:
        """
        # Create positions DataFrame with the same size as input signal
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        # Calculate position with fixed size
        positions[self.symbol] = self.purchasing_size*self.signals['signal']
        return positions

    def backtest_portfolio(self):
        """
        backtesting portfolio with the generated positions
        :return:
        """
        # Create portfolio DataFrame
        pos_diff = self.positions.diff()
        self.portfolio = self.positions*self.bars
        self.portfolio['stock'] = (self.positions*self.bars).sum(axis=1)
        self.portfolio['cash'] = self.initial_capital - (pos_diff*self.bars).sum(axis=1).cumsum()
        self.portfolio['portfolio'] = self.portfolio['cash'] + self.portfolio['stock']
        self.portfolio['returns'] = self.portfolio['portfolio'].pct_change()

        return self.portfolio

    def calculate_sharpe_ratio(self, bmk):
        """
        calculate Sharpe Ratio against input benchmark
        :param bmk:
        :return:
        """
        bmk_table = pd.DataFrame([])
        bmk_table['bmk'] = bmk
        bmk_table['returns'] = bmk_table['bmk'].pct_change().fillna(0.0)
        self.portfolio['bmk_returns'] = pd.DataFrame(bmk_table['returns'],
                                                     index=self.portfolio.index).fillna(0.0)
        self.portfolio['excess_return'] = self.portfolio["returns"] - self.portfolio['bmk_returns']
        sharpe_ratio = self.portfolio['excess_return'].mean() / \
                       self.portfolio['excess_return'].std() * math.sqrt(252)
        return sharpe_ratio


