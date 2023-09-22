import numpy as np


class Environment4():
    def __init__(self, n_arms, probabilities, features):
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.features = features
        self.n1 = lambda x: 65 * (1 - np.exp(-3.6 * x + 2 * x ** 3)) + 2
        self.n2 = lambda x: 65 * (1 - np.exp(-3.2 * x + 2 * x ** 3)) + 2
        self.n3 = lambda x: 24 * (1 - np.exp(-3 * x + x ** 3)) + 1.5
        self.n4 = lambda x: 24 * (1 - np.exp(-3 * x + x ** 3)) + 1.5
        self.cc1 = lambda x: 100 * (1 - np.exp(-4.5 * x + x ** 3)) + 11
        self.cc2 = lambda x: 95 * (1 - np.exp(-4 * x + x ** 3)) + 10
        self.cc3 = lambda x: 45 * (1 - np.exp(-3 * x + x ** 3)) + 5
        self.cc4 = lambda x: 45 * (1 - np.exp(-3 * x + x ** 3)) + 5

    def n(self, x):
        if self.features == [0, 0]:
            return self.n1(x)
        elif self.features == [0, 1]:
            return self.n2(x)
        elif self.features == [1, 0]:
            return self.n3(x)
        elif self.features == [1, 1]:
            return self.n4(x)

    def cc(self, x):
        if self.features == [0, 0]:
            return self.cc1(x)
        elif self.features == [0, 1]:
            return self.cc2(x)
        elif self.features == [1, 0]:
            return self.cc3(x)
        elif self.features == [1, 1]:
            return self.cc4(x)

    def draw_n(self, x, noise_std):
        if self.features == "00" or self.features == [0, 0]:
            return self.n1(x) + np.random.normal(0, noise_std, size=self.n(x).shape)
        if self.features == "01" or self.features == [0, 1]:
            return self.n2(x) + np.random.normal(0, noise_std, size=self.n(x).shape)
        if self.features == "10" or self.features == [1, 0]:
            return self.n3(x) + np.random.normal(0, noise_std, size=self.n(x).shape)
        if self.features == "11" or self.features == [1, 1]:
            return self.n4(x) + np.random.normal(0, noise_std, size=self.n(x).shape)

    def draw_cc(self, x, noise_std):
        if self.features == "00" or self.features == [0, 0]:
            return self.cc1(x) + np.random.normal(0, noise_std, size=self.n(x).shape)
        if self.features == "01" or self.features == [0, 1]:
            return self.cc2(x) + np.random.normal(0, noise_std, size=self.n(x).shape)
        if self.features == "10" or self.features == [1, 0]:
            return self.cc3(x) + np.random.normal(0, noise_std, size=self.n(x).shape)
        if self.features == "11" or self.features == [1, 1]:
            return self.cc4(x) + np.random.normal(0, noise_std, size=self.n(x).shape)

    def draw_n2(self, x, noise_std):
        if self.features == [None, None]:
            return self.n1(x) + self.n2(x) + self.n3(x) + self.n4(x) + np.random.normal(0, noise_std,
                                                                                        size=self.n1(x).shape)
        elif self.features == [None, 0]:
            return self.n1(x) + self.n3(x) + np.random.normal(0, noise_std / 2, size=self.n1(x).shape)
        elif self.features == [None, 1]:
            return self.n2(x) + self.n4(x) + np.random.normal(0, noise_std / 2, size=self.n1(x).shape)
        elif self.features == [0, None]:
            return self.n1(x) + self.n2(x) + np.random.normal(0, noise_std / 2, size=self.n1(x).shape)
        elif self.features == [1, None]:
            return self.n3(x) + self.n4(x) + np.random.normal(0, noise_std / 2, size=self.n1(x).shape)
        elif self.features == [0, 0]:
            return self.n1(x) + np.random.normal(0, noise_std / 4, size=self.n1(x).shape)
        elif self.features == [0, 1]:
            return self.n2(x) + np.random.normal(0, noise_std / 4, size=self.n1(x).shape)
        elif self.features == [1, 0]:
            return self.n3(x) + np.random.normal(0, noise_std / 4, size=self.n1(x).shape)
        elif self.features == [1, 1]:
            return self.n4(x) + np.random.normal(0, noise_std / 4, size=self.n1(x).shape)
        else:
            return 0

    def draw_cc2(self, x, noise_std):
        if self.features == [None, None]:
            return self.cc1(x) + self.cc2(x) + self.cc3(x) + self.cc4(x) + np.random.normal(0, noise_std,
                                                                                            size=self.cc1(x).shape)
        elif self.features == [None, 0]:
            return self.cc1(x) + self.cc3(x) + np.random.normal(0, noise_std / 2, size=self.cc1(x).shape)
        elif self.features == [None, 1]:
            return self.cc2(x) + self.cc4(x) + np.random.normal(0, noise_std / 2, size=self.cc1(x).shape)
        elif self.features == [0, None]:
            return self.cc1(x) + self.cc2(x) + np.random.normal(0, noise_std / 2, size=self.cc1(x).shape)
        elif self.features == [1, None]:
            return self.cc3(x) + self.cc4(x) + np.random.normal(0, noise_std / 2, size=self.cc1(x).shape)
        elif self.features == [0, 0]:
            return self.cc1(x) + np.random.normal(0, noise_std / 4, size=self.cc1(x).shape)
        elif self.features == [0, 1]:
            return self.cc2(x) + np.random.normal(0, noise_std / 4, size=self.cc1(x).shape)
        elif self.features == [1, 0]:
            return self.cc3(x) + np.random.normal(0, noise_std / 4, size=self.cc1(x).shape)
        elif self.features == [1, 1]:
            return self.cc4(x) + np.random.normal(0, noise_std / 4, size=self.cc1(x).shape)
        else:
            return 0

    def round(self, pulled_arm, user_features):  # given an arm, return a reward
        if self.features == "00" or self.features == [0, 0]:
            return np.random.binomial(1, self.probabilities[pulled_arm, 0])
        elif self.features == "01" or self.features == [0, 1]:
            return np.random.binomial(1, self.probabilities[pulled_arm, 1])
        elif self.features == "10" or self.features == [1, 0]:
            return np.random.binomial(1, self.probabilities[pulled_arm, 2])
        elif self.features == "11" or self.features == [1, 1]:
            return np.random.binomial(1, self.probabilities[pulled_arm, 3])
        else:
            return 0
