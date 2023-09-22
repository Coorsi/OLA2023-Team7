import numpy as np


class Environment():
    def __init__(self, n_arms, probabilities,
                 id_class):  # probabilities is the probability distribution of the arm rewards
        self.n_arms = n_arms
        self.probabilities = probabilities
        self.id_class = id_class

    def n(self, x):
        if self.id_class == 0:
            return 65 * (1 - np.exp(-3.6 * x + 2 * x ** 3)) + 2
        elif self.id_class == 1:
            return 65 * (1 - np.exp(-3.2 * x + 2 * x ** 3)) + 2
        elif self.id_class == 2:
            return 48 * (1 - np.exp(-3 * x + x ** 3)) + 3
        else:
            return -1

    def cc(self, x):
        if self.id_class == 0:
            return 100 * (1 - np.exp(-4.5 * x + x ** 3)) + 11
        elif self.id_class == 1:
            return 95 * (1 - np.exp(-4 * x + x ** 3)) + 10
        elif self.id_class == 2:
            return 90 * (1 - np.exp(-3 * x + x ** 3)) + 10
        else:
            return -1

    def draw_n(self, x, noise_std):
        return self.n(x) + np.random.normal(0, noise_std, size=self.n(x).shape)

    def draw_cc(self, x, noise_std):
        return self.cc(x) + np.random.normal(0, noise_std, size=self.cc(x).shape)

    def round(self, pulled_arm):  # given an arm, return a reward
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward

    def reward(self, conv_rate, bid, margin):
        return (self.n(bid) * conv_rate * margin - self.cc(bid))


class Non_Stationary_Environment(Environment):
    def __init__(self, n_arms, probabilities, id_class, horizon, high_frequency_change):
        super().__init__(n_arms, probabilities, id_class)
        self.time = 0
        self.high_frequency_change = high_frequency_change
        self.n_phases = len(self.probabilities)
        self.current_phase = 0
        if self.high_frequency_change == 0:
            self.phases_size = int(horizon / self.n_phases)
        else:
            self.phases_size = int(horizon / (self.n_phases * 4))

    def round2(self, pulled_arm, n, update_time):    # pulled arm, number of clicks, bool (if true update time)
        if self.high_frequency_change == 0:
            if self.time % self.phases_size == 0 and self.time != 0 and update_time:
                if self.current_phase == 2:
                    self.current_phase = 0
                else:
                    self.current_phase += 1
        else:
            if self.time % 18 == 0 and self.time != 0 and update_time:
                if self.current_phase == 4:
                    self.current_phase = 0
                else:
                    self.current_phase += 1
        p = self.probabilities[self.current_phase][pulled_arm]
        reward = 0
        for i in range(n):
            reward += np.random.binomial(1, p)
        if update_time:
            self.time += 1
        return reward
