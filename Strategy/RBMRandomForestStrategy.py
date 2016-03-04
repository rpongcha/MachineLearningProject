import Utils as ut
import models.random_forest as rb
import glob
import os
from BaseStrategy import BaseStrategy
import datetime

class RBMRandomForestStrategy(BaseStrategy):
    def __init__(self, symbol, data_point, i):
        """
        Constructor of the class
        :param symbol: symbol name
        :param bars: historical prices
        :param short_window: short period looking back in time
        :param long_window: long period looking back in time
        :return: N/A
        """
        self.symbol = symbol
        self.data_point = data_point
        self.iteration = i
        data = ut.read_price_from_csv(symbol, self.data_point, i)
        self.__time__ = [ut.convert_to_time(i) for i in data[0]]
        self.__bars__ = data[1]
        # read from file
        self.signals = []

    def generate_signals(self):
        """
        Generate signal according input file
        :return: signals
        """
        filename = os.path.join(os.path.expanduser('~'), 'RA', 'MachineLearningProject', 'data','random_forest', self.symbol+'_'+str(self.iteration)+'*')
        filenames = glob.glob(filename)
        if len(filenames) < 1:
            rb.process_machine_learning(self.symbol, self.iteration)

        self.signals = ut.read_signals_from_file(self.symbol, self.data_point,self.iteration)
        # read from file

        return self.signals[:,0],self.signals[:,1]

    def get_time_stamp(self):
        return self.__time__

    def get_prices(self):
        return self.__bars__



