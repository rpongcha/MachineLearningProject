import csv
import glob
import numpy as np


def read_price_from_csv(key, data_point, bottom=True):
    prices = []
    filename = 'data/'+key+'*.csv'
    filenames = glob.glob(filename)

    if len(filenames) >= 1:
        filename = filenames[0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = list(reader)
            if bottom:
                start = len(rows) - data_point
                prices = np.array(rows[start:])
            else:
                prices = np.array(rows[0:data_point])
            prices = np.transpose(prices)

    return prices


def read_signals_from_file(key, data_point):
    signals = np.zeros((data_point, 2))

    filename = 'PRED_'+key+'*.csv'
    filenames = glob.glob(filename)

    if len(filenames) >= 1:
        filename = filenames[0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = list(reader)
            for i in range(len(rows)):
                if i < data_point:
                    signals[i, 0] = float(rows[i][0])
                    signals[i, 1] = float(rows[i][1])

    return signals

if __name__ == '__main__':
    print(read_price_from_csv('AD', 12500))
    print(read_signals_from_file('AD', 12500))
