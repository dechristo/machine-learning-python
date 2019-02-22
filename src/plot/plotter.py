import matplotlib.pyplot as plot
import numpy as np
from src.utils.csv_reader import CsvReader
from src.neural_network.perceptron import Perceptron
from matplotlib.pyplot import figure
figure(num=None, figsize=(18, 7), dpi=80, facecolor='w', edgecolor='k')

class Plotter:

    @staticmethod
    def iris_test():
        iris_data_set = CsvReader.read()

        # select setosa and versicolor
        y = iris_data_set.iloc[0:100, 4].values
        y = np.where(y == 'learning_rate', -1, 1)

        # extract sepal length and petal length
        X = iris_data_set.iloc[0:100, [0, 2]].values

        # plot data
        plot.subplot(1, 2, 1)
        plot.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label = 'setosa')
        plot.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker = 'x', label = 'versicolor')
        plot.xlabel('sepal length [cm]')
        plot.ylabel('petal length [cm]')
        plot.legend(loc='upper left')

        ppn = Perceptron(learning_rate=0.1, epochs=10)
        ppn.fit(X, y)

        plot.subplot(1,2,2)
        plot.plot(range(1, len(ppn.misclassifications) + 1), ppn.misclassifications, marker = 'o')
        plot.xlabel('Epochs')
        plot.ylabel('Number of updates')
        plot.show()