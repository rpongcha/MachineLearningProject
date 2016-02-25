import Utils as ut
import models.rbm_random_forest as rb
import glob
import os
from BaseStrategy import BaseStrategy


class RBMRandomForestStrategy(BaseStrategy):
    def __init__(self, symbol, data_point):
        """
        Constructor of the class
        :param symbol: symbol name
        :param data_point: how many data points used for testing
        :return: N/A
        """
        self.symbol = symbol
        self.data_point = data_point
        # read data from csv file
        data = ut.read_price_from_csv(symbol, self.data_point)
        # assign time stamp
        self.__time__ = [ut.convert_to_time(i) for i in data[:, 0]]
        # self.__time__ = data[:, 0]
        # assign prices
        self.__bars__ = data[:, 1]
        # read from file
        self.signals = []

    def generate_signals(self):
        """
        Generate signal according input file
        :return: signals
        """
        # Check if the csv file does not exist, perform machine learning
        filename = 'PRED_'+self.symbol+'*.csv'
        # filename = os.path.join(os.path.expanduser('~'), 'RA', 'python_machine_learning',
        #                        'src', 'models', self.symbol+'*')
        filenames = glob.glob(filename)
        if len(filenames) < 1:
            rb.process_machine_learning()

        # Read signals from .csv file
        self.signals = ut.read_signals_from_file(self.symbol, self.data_point)[:, 0]

        return self.signals

    def get_time_stamp(self):
        return self.__time__

    def get_prices(self):
        return self.__bars__



