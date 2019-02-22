import pandas as pd


class CsvReader:

    @staticmethod
    def read():
        return pd.read_csv('data/iris.csv', header=None)