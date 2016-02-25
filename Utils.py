import csv
import glob
import numpy as np
import datetime as dt
import time as tm


def read_price_from_csv(key, data_point, bottom=True):
    prices = np.zeros((data_point, 2))
    filename = 'data/'+key+'*.csv'
    filenames = glob.glob(filename)

    if len(filenames) >= 1:
        filename = filenames[0]
        with open(filename, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = list(reader)

            start = len(rows) - data_point if bottom else 0
            end = len(rows) if bottom else data_point
            cnt = 0
            for i in range(start, end):
                if cnt < data_point:
                    prices[cnt, 0] = float(rows[i][0])
                    prices[cnt, 1] = float(rows[i][1])
                    cnt += 1

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


def file_exists(directory, key_name):
    full_filename = directory+"/"+key_name
    filenames = glob.glob(full_filename)
    exists = (len(filenames) >= 1)
    return exists


def convert_to_time(timestamp):
    temp = dt.timedelta(timestamp)
    time = dt.datetime(0001, 1, 1) + dt.timedelta(days=temp.days) + dt.timedelta(seconds=temp.seconds)
    # time = tm.gmtime(dt.timedelta(timestamp).seconds)
    return time.strftime("%Y/%m/%d %H:%M:%S.%f")

if __name__ == '__main__':
    # print(read_price_from_csv('AD', 12500))
    # print(read_signals_from_file('AD', 12500))
    print(convert_to_time(726835.600694444))
