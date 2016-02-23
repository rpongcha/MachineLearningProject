import csv
import glob
import numpy as np


def read_price_from_csv(key, length, bottom=True):
    prices = []
    filename = 'data/'+key+'*.csv'
    filenames = glob.glob(filename)

    if len(filenames) >= 1:
        filename = filenames[0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = list(reader)
            if bottom:
                start = len(rows) - length + 1
                prices = np.array(rows[start:])
            else:
                prices = np.array(rows[0:length - 1])
            prices = np.transpose(prices)

    return prices


def read_signals_from_file(key):
    signals = []

    filename = 'models/PRED_'+key+'*.csv'
    filenames = glob.glob(filename)

    if len(filenames) >= 1:
        filename = filenames[0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = list(reader)
            signals = np.transpose(np.array(rows))

    return signals

if __name__ == '__main__':
    print(read_price_from_csv('AD', 12500))
    print(read_signals_from_file('AD'))
