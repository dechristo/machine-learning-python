import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=10, random_weight=1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_weight = random_weight
        self.weights = []
        self.misclassifications = []

    def fit(self, training_array, target_array):
        random_gen = np.random.RandomState(self.random_weight)
        self.weights = random_gen.normal(loc=0.0, scale=0.01, size=1 + training_array.shape[1])
        for index in range(self.epochs):
            miss = 0
            for training_samples, target in zip(training_array, target_array):
                new_value = self.learning_rate * (target - self.predict(training_samples))
                self.weights[1:] += new_value * training_samples
                self.weights[0] += new_value
                miss += int(new_value != 0.0)
            self.misclassifications.append(miss)
        return self

    def neural_network_input(self, training_samples):
        return np.dot(training_samples, self.weights[1:]) + self.weights[0]

    def predict(self, training_samples):
        return np.where(self.neural_network_input(training_samples) >= 0.0, 1, -1)

