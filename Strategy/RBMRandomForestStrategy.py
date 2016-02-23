import pandas as pd
import numpy as np
import Utils as ut

from BaseStrategy import BaseStrategy


class RBMRandomForestStrategy(BaseStrategy):
    def __init__(self, symbol, datapoint):
        """
        Constructor of the class
        :param symbol: symbol name
        :param bars: historical prices
        :param short_window: short period looking back in time
        :param long_window: long period looking back in time
        :return: N/A
        """
        self.symbol = symbol
        data = ut.read_price_from_csv(symbol, datapoint)
        self.time = data[0]
        self.bars = data[1]
        # read from file
        self.signals = []

    def generate_signals(self):
        """
        Generate signal according input file
        :return: signals
        """
        self.signals = ut.read_signals_from_file(self.symbol)[0]
        # read from file

        return self.signals



