import csv
import glob


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
                start = len(rows) - length
                prices = rows[start:]
            else:
                prices = rows[0:length]

    return prices


if __name__ == '__main__':
    read_price_from_csv('AD', 12500)
