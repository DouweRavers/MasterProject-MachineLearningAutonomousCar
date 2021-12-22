import csv
import numpy as np


def loadDataFile(limiter=-1):
    data = []
    row_count = 0
    with open('data/data.csv', "r") as csvDataFile:
        csvReader = csv.reader(csvDataFile, delimiter=",")
        for row in csvReader:
            if row[0] == "input":
                continue
            row = [value.replace(",", ".") for value in row]
            data.append(row)
            if limiter == row_count:
                break
            row_count += 1
    np.random.shuffle(data)
    data = np.array(data)
    Y = data[:, 0]
    Y = np.array(Y).astype(float)
    Y = Y * 0.5 + 0.5
    X = data[:, 1:]
    X = np.array(X).astype(float)
    return X, Y
