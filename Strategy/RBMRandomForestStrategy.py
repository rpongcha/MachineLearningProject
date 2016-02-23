import Utils as ut
import models.rbm_random_forest as rb
import glob
from BaseStrategy import BaseStrategy


class RBMRandomForestStrategy(BaseStrategy):
    def __init__(self, symbol, data_point):
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
        data = ut.read_price_from_csv(symbol, self.data_point)
        self.__time__ = data[0]
        self.__bars__ = data[1]
        # read from file
        self.signals = []

    def generate_signals(self):
        """
        Generate signal according input file
        :return: signals
        """
        filename = 'PRED_'+self.symbol+'*.csv'
        filenames = glob.glob(filename)
        if len(filenames) < 1:
            rb.process_machine_learning()

        self.signals = ut.read_signals_from_file(self.symbol, self.data_point)[:, 0]
        # read from file

        return self.signals

    def get_time_stamp(self):
        return self.__time__

    def get_prices(self):
        return self.__bars__



