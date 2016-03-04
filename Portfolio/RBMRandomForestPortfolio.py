from BasePortfolio import BasePortfolio
import pandas as pd
import math
import numpy as np


class RBMRandomForestPortfolio(BasePortfolio):

    def __init__(self, symbol, prices, signals, initial_capital=100000.0, initial_margin=5000,maint_margin=3500, contract_size = 1000, purchase_size=1.):
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
        self.prices = prices
        self.length = len(prices)
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.initial_margin = float(initial_margin)
        self.maint_margin = maint_margin
        self.contract_size = contract_size
        # the purchasing size is fixed at 100 shares
        self.purchasing_size = purchase_size
        # generate positions from the input signals
        
        self.portfolio = pd.DataFrame({'prices':self.prices['prices']},index = self.prices.index)
        
        self.portfolio['prices_change']=self.portfolio['prices'].diff().fillna(0)
        self.portfolio['signals']=self.signals

    def generate_positions(self):
        """
        generate positions from the signals
        :return:
        """
        # Create positions DataFrame with the same size as input signal
        #positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        # Calculate position with fixed size
        #positions[self.symbol] = self.purchasing_size*self.signals['signal']
        #return positions
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
                account[i]=account[i-1]+curr_pos*self.portfolio['prices_change'][i]*self.contract_size

            #if a buying signal, need to have account able to pay additional initial margin
            if self.portfolio['signals'][i]>0:
                if account[i]>(curr_margin+self.portfolio['signals'][i]*self.purchasing_size*self.initial_margin):
                    curr_pos+=self.portfolio['signals'][i]*self.purchasing_size
                    curr_margin=curr_pos*self.maint_margin
            #if a selling signal, need to have previously purchase of the futures
            elif ((self.portfolio['signals'][i]<0) and (curr_pos >0)):
                curr_pos+=self.portfolio['signals'][i]*self.purchasing_size
                curr_margin=curr_pos*self.maint_margin
        self.portfolio['portfolio']=account
        #self.portfolio['stock'] = (self.positions*self.bars).sum(axis=1)
        #self.portfolio['cash'] = self.initial_capital - (pos_diff*self.bars).sum(axis=1).cumsum()
        #self.portfolio['portfolio'] = self.portfolio['cash'] + self.portfolio['stock']
        self.portfolio['returns'] = self.portfolio['portfolio'].pct_change().fillna(0)
        self.portfolio['P&L'] = self.portfolio['portfolio'].diff().fillna(0)
        self.portfolio['Cumulative P&L'] = self.portfolio['P&L'] .cumsum()
        return self.portfolio

    def calculate_sharpe_ratio(self):
        """
        calculate Sharpe Ratio against input benchmark
        :param bmk:
        :return:
        """
        sharpe_ratio = self.portfolio['returns'].mean() / \
                       self.portfolio['returns'].std() * math.sqrt(252)
        return sharpe_ratio


