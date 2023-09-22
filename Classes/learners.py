import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import math
import random
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)


class Learner():
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.t = 0
        self.reward_per_arm = [[] for i in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):  # update the observation list once the reward is returned
        self.reward_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)


class TS_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))

    def pull_arm(self, margins):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * margins)
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[2])
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward[0]
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + reward[1]


class UCB1_Learner(Learner):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.empirical_means = np.zeros(n_arms)
        self.confidence = np.array([np.inf] * n_arms)
        self.n_pulled = np.zeros(n_arms)  # how many times an arm was pulled

    def pull_arm(self, margins):
        upper_conf = (self.empirical_means + self.confidence) * margins
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        if reward[0] != 0 or reward[1] != 0 or self.n_pulled[pulled_arm] != 0:
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * self.n_pulled[pulled_arm] + reward[
                0]) / (
                                                       self.n_pulled[pulled_arm] + reward[0] + reward[1])
        # self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm]*(self.t-1) + reward[0])/(self.t + reward[0] + reward[1])
        for a in range(self.n_arms):
            self.confidence[a] = (2 * np.log(self.t) / self.n_pulled[a]) ** 0.5 if self.n_pulled[a] > 0 else np.inf
        self.update_observations(pulled_arm, reward[2])
        self.n_pulled[pulled_arm] += reward[0] + reward[1]


class SWTS_Learner(TS_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])

    def pull_arm(self, margins):
        idx = np.argmax(np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1]) * margins)
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[2])
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            cum_rew = np.sum(self.reward_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            self.beta_parameters[arm, 0] = cum_rew + reward[0]
            self.beta_parameters[arm, 1] = n_samples - cum_rew + reward[1]


class SWUCB_Learner(UCB1_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])
        self.success_round = np.array([])    # number of conversions per round
        self.samples_round = np.array([])    # number of clicks per round

    def pull_arm(self, margins):
        upper_conf = (self.empirical_means + self.confidence) * margins
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[2])
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        self.samples_round = np.append(self.samples_round, reward[1] + reward[0])
        self.success_round = np.append(self.success_round, reward[0])
        for arm in range(self.n_arms):
            n_samples = np.sum((self.pulled_arms[-self.window_size:] == arm) * self.samples_round[-self.window_size:])
            cum_rew = np.sum((self.pulled_arms[-self.window_size:] == arm) * self.success_round[-self.window_size:])
            self.empirical_means[arm] = cum_rew / n_samples if n_samples > 0 else 0
            self.confidence[arm] = (2 * np.log(self.t) / n_samples) ** 0.5 if n_samples > 0 else np.inf


class GPTS_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms) * 10
        self.pulled_arms = []
        alpha = 1
        kernel = C(1e1, (1e-7, 1e7)) * RBF(1e1, (1e-10, 1e7))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2)
        self.iteration = 0
        self.refitFrequency = 13  # 13 bc it divides 364

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        self.iteration += 1
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        if self.iteration % self.refitFrequency == 0:
            self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        sampled_values = np.random.normal(self.means, self.sigmas)
        return np.argmax(sampled_values)


class GPUCB_Learner(Learner):
    def __init__(self, n_arms, arms):
        super().__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms) * 10
        self.pulled_arms = []
        alpha = 1
        kernel = C(1e1, (1e-7, 1e7)) * RBF(1e1, (1e-10, 1e7))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha ** 2, n_restarts_optimizer=5)
        self.iteration = 0
        self.refitFrequency = 13  # 13 bc it divides 364

    def update_observations(self, pulled_arm, reward):
        super().update_observations(pulled_arm, reward)
        self.pulled_arms.append(self.arms[pulled_arm])

    def update_model(self):
        self.iteration += 1
        x = np.atleast_2d(self.pulled_arms).T
        y = self.collected_rewards
        if self.iteration % self.refitFrequency == 0:
            self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.means = np.array(self.means)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arm(self):
        upper_conf = self.means + 1.96 * self.sigmas
        return np.random.choice(np.where(upper_conf == upper_conf.max())[0])


class CUSUM:
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample / self.M
            return 0
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t = 0
        self.g_minus = 0
        self.g_plus = 0
        self.reference = 0


class CUSUM_UCB_Learner(UCB1_Learner):
    def __init__(self, n_arms, M=100, eps=0.05, h=20, alpha=0.01):
        super().__init__(n_arms)
        self.n_arms = n_arms
        self.change_detection = [CUSUM(M, eps, h) for _ in range(n_arms)]
        self.valid_rewards_per_arm = [0 for _ in range(n_arms)]
        self.valid_samples_per_arm = [0 for _ in range(n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.valid_round_per_arm = [0 for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self, margins):
        if np.random.binomial(1, 1 - self.alpha):
            upper_conf = (self.empirical_means + self.confidence) * margins
            return np.random.choice(np.where(upper_conf == upper_conf.max())[0])
        else:
            return np.random.randint(0, self.n_arms)

    def update(self, pulled_arm, reward):
        self.t += 1
        flag = 0
        # I need to feed cusum every single success/failure

        for i in range(reward[0]):
            if self.change_detection[pulled_arm].update(1):
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arm[pulled_arm] = 0
                self.valid_samples_per_arm[pulled_arm] = 0
                self.valid_round_per_arm[pulled_arm] = 0
                self.change_detection[pulled_arm].reset()
                flag = 1
        if not flag:
            for i in range(reward[1]):
                if self.change_detection[pulled_arm].update(0):
                    self.detections[pulled_arm].append(self.t)
                    self.valid_rewards_per_arm[pulled_arm] = 0
                    self.valid_samples_per_arm[pulled_arm] = 0
                    self.valid_round_per_arm[pulled_arm] = 0
                    self.change_detection[pulled_arm].reset()

        flag = 0

        self.valid_round_per_arm[pulled_arm] += 1
        self.update_observations(pulled_arm, reward)
        self.empirical_means[pulled_arm] = self.valid_rewards_per_arm[pulled_arm] / self.valid_samples_per_arm[
            pulled_arm] if self.valid_samples_per_arm[pulled_arm] > 0 else 0
        total_valid_samples = np.sum(self.valid_round_per_arm)
        for a in range(self.n_arms):
            self.confidence[a] = (2 * np.log(total_valid_samples) / self.valid_samples_per_arm[a]) ** 0.5 if \
            self.valid_samples_per_arm[a] > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.valid_rewards_per_arm[pulled_arm] += reward[0]
        self.valid_samples_per_arm[pulled_arm] += reward[0] + reward[1]
        self.reward_per_arm[pulled_arm].append(reward[2])
        self.collected_rewards = np.append(self.collected_rewards, reward[2])


class EXP3_Learner(Learner):
    def __init__(self, n_arms, gamma):
        super().__init__(n_arms)
        self.gamma = gamma
        self.weights = [1.0] * n_arms
        self.arm_index = [i for i in range(n_arms)]
        self.probabilityDistribution = []

    def distr(self):
        theSum = float(sum(self.weights))
        return tuple((1.0 - self.gamma) * (w / theSum) + (self.gamma / len(self.weights)) for w in self.weights)

    def draw(self, weights):
        choice = random.uniform(0, sum(self.weights))
        choiceIndex = 0

        for weight in self.weights:
            choice -= weight
            if choice <= 0:
                return choiceIndex

            choiceIndex += 1

    def pull_arm(self):
        self.probabilityDistribution = self.distr()
        idx = self.draw(self.probabilityDistribution)
        return idx

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward[2])
        if reward[2] < 0:
            scaledRew = 0
        else:
            scaledRew = reward[2] / 320
        estimatedReward = 1.0 * scaledRew / self.probabilityDistribution[pulled_arm]
        self.weights[pulled_arm] *= math.exp(estimatedReward * self.gamma / self.n_arms)
