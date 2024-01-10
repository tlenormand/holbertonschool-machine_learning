#!/usr/bin/env python3
""" preprocesses data """

import numpy as np


processed_data_data_format = {
    "timestamp": int,
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume_(btc)": float,
    "volume_(currency)": float,
    "weighted_price": float
}


def fill_values(bs, i, cb):
    """ fill values """
    prev_line = bs[i - 1]
    j = i

    while np.isnan(bs[j][1]):
        j += 1

    next_line = bs[j]

    # line values
    for k in range(i, j):
        bs[k] = prev_line + (next_line - prev_line) * (k - i) / (j - i)

    # timestamp
    for k in range(i, j):
        bs[k][0] = int(bs[k - 1][0] + 60)

    return bs


def fill_NaN(bs, cb):
    """ fill NaN lines """
    for i in range(len(bs)):
        if np.isnan(bs[i][1]):
            bs = fill_values(bs, i, cb)

    return bs


def mean_by_hours(bs):
    """ mean of 24h """
    for i in range(len(bs)):
        if i < 60:
            bs[i] = np.mean(bs[:i + 1, 1])
        else:
            bs[i] = np.mean(bs[i - 60 + 1:i + 1, 1])

    return bs


def preprocess_data(file_path=None):
    """ preprocesses data """
    if file_path is None:
        bitstamp = './data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
        # coinbase = './data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'

        bs = np.genfromtxt(bitstamp, delimiter=',', skip_header=1)
        # cb = np.genfromtxt(coinbase, delimiter=',', skip_header=1)

        full_data = fill_NaN(bs, None)
        # full_data = mean_by_hours(full_data)

        np.savetxt("./data/full_data.csv", full_data, delimiter=",")
    else:
        full_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)

    count = 0
    for i in range(len(full_data)):
        if np.isnan(full_data[i][1]):
            print("INFO: NaN found in at index {}".format(i))
            count += 1

    if count > 0:
        print("ERROR: {} NaNs found".format(count))
        exit(-1)

    return full_data


if __name__ == '__main__':
    # generate full_data.csv
    bs = preprocess_data()
