from BasePortfolio import BasePortfolio
import pandas as pd
import math
import numpy as np


class RBMRandomForestPortfolio(BasePortfolio):

    def __init__(self, symbol, prices, signals, initial_capital=100000.0, initial_margin=5000,maint_margin=3500, purchase_size=1.):
        """
        Construct of the class
        :param symbol: 2 alphabets symbol
        :param bars: futures contract prices
        :param signals: BUY/SELL/HOLD signals
        :param initial_capital: initial capital
        :param initial_margin: initial margin required per contract
        :param maint_margin: maintenance margin required per contract
        :return:
        """
        self.symbol = symbol
        self.prices = prices
        self.length = len(prices)
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.initial_margin = float(initial_margin)
        self.maint_margin = maint_margin
        # the purchasing size is fixed at 1 shares
        self.purchasing_size = purchase_size

        # generate positions from the input signals
        self.portfolio = pd.DataFrame({'prices': self.prices['prices']}, index=self.prices.index)
        self.portfolio['prices_change'] = self.portfolio['prices'].diff().fillna(0)
        self.portfolio['signals'] = self.signals

    def generate_positions(self):
        """
        generate positions from the signals
        :return:
        """
        # Create positions DataFrame with the same size as input signal
        # positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        # Calculate position with fixed size
        # positions[self.symbol] = self.purchasing_size*self.signals['signal']
        # return positions
        return 0
    def backtest_portfolio(self):
        """
        backtesting portfolio with the generated positions
        :return:
        """
        # Create portfolio DataFrame
        position = np.zeros(self.length)
        account = np.ones(self.length)*self.initial_capital
        curr_pos = 0
        curr_margin = 0
        for i in range(self.length):
            if i !=0:
                account[i]=account[i-1]+curr_pos*self.portfolio['prices_change'][i]
            if account[i]<curr_margin:
                # close out position when account value less than margin requirement
                account[i:]=account[i]
                break
            else:
                # if a buying signal, need to have account able to pay additional initial margin
                if self.portfolio['signals'][i]>0:
                    if account[i]>(curr_margin+self.portfolio['signals'][i]*self.purchasing_size*self.initial_margin):
                        curr_pos+=self.portfolio['signals'][i]*self.purchasing_size
                        curr_margin=curr_pos*self.maint_margin
                # if a selling signal, need to have previously purchase of the futures
                elif ((self.portfolio['signals'][i]<0) and (curr_pos >0)):
                    curr_pos+=self.portfolio['signals'][i]*self.purchasing_size
                    curr_margin=curr_pos*self.maint_margin
        self.portfolio['portfolio']=account
        # self.portfolio['stock'] = (self.positions*self.bars).sum(axis=1)
        # self.portfolio['cash'] = self.initial_capital - (pos_diff*self.bars).sum(axis=1).cumsum()
        # self.portfolio['portfolio'] = self.portfolio['cash'] + self.portfolio['stock']
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


