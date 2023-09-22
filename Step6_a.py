import math

from Classes.learners import SWUCB_Learner, CUSUM_UCB_Learner, \
    EXP3_Learner
from Classes.enviroment import Non_Stationary_Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

n_prices = 5
n_bids = 100
cost_of_product = 180
price = 100

bids = np.linspace(0.0, 1.0, n_bids)
prices = price * np.array([1, 2, 3, 4, 5])
margins = np.array([prices[i] - cost_of_product for i in range(n_prices)])
classes = np.array([0, 1, 2])

conversion_rate_phase1 = np.array([[0.69, 0.29, 0.38],  # 1*price
                                   [0.57, 0.23, 0.31],  # 2*price
                                   [0.51, 0.18, 0.24],  # 3*price
                                   [0.11, 0.12, 0.17],  # 4*price
                                   [0.05, 0.07, 0.09]  # 5*price
                                   ])
# C1   C2   C3
conversion_rate_phase2 = np.array([[0.46, 0.18, 0.25],  # 1*price
                                   [0.21, 0.15, 0.20],  # 2*price
                                   [0.15, 0.11, 0.15],  # 3*price
                                   [0.12, 0.06, 0.10],  # 4*price
                                   [0.04, 0.03, 0.05]  # 5*price
                                   ])

# C1   C2   C3
conversion_rate_phase3 = np.array([[0.77, 0.48, 0.70],  # 1*price
                                   [0.51, 0.37, 0.56],  # 2*price
                                   [0.44, 0.28, 0.42],  # 3*price
                                   [0.37, 0.14, 0.28],  # 4*price
                                   [0.34, 0.05, 0.14]  # 5*price
                                   ])

env_array = []
T = 365
for c in classes:
    env_array.append(Non_Stationary_Environment(n_prices, np.array(
        [conversion_rate_phase1[:, c], conversion_rate_phase2[:, c], conversion_rate_phase3[:, c]]), c, T, 0))

# EXPERIMENT BEGIN

n_experiments = 100

M = 100  # number of steps to obtain reference point in change detection (for CUSUM)
eps = 0.1  # epsilon for deviation from reference point in change detection (for CUSUM)
h = np.log(T) ** 2  # threshold for change detection (for CUSUM)

swucb_rewards_per_experiments = []
cusum_rewards_per_experiments = []
exp3low_rewards_per_experiments = []
exp3mid_rewards_per_experiments = []
exp3high_rewards_per_experiments = []

optimal1 = clairvoyant(classes, bids, prices, margins, conversion_rate_phase1, env_array)
opt_index_phase1 = optimal1[0][0]
opt_phase1 = optimal1[2][0]
optimal_bid_index_phase1 = optimal1[1][0]
optimal_bid_phase1 = bids[int(optimal_bid_index_phase1)]  # we consider the same bid (?)

optimal2 = clairvoyant(classes, bids, prices, margins, conversion_rate_phase2, env_array)
opt_index_phase2 = optimal2[0][0]
opt_phase2 = optimal2[2][0]

optimal3 = clairvoyant(classes, bids, prices, margins, conversion_rate_phase3, env_array)
opt_index_phase3 = optimal2[0][0]
opt_phase3 = optimal3[2][0]

for e in tqdm(range(n_experiments)):
    env = deepcopy(env_array[0])
    swucb_learner = SWUCB_Learner(n_arms=n_prices, window_size=(int(math.sqrt(T)) * 6))
    cusum_learner = CUSUM_UCB_Learner(n_arms=n_prices, M=M, eps=eps, h=h)
    exp3_learner_low = EXP3_Learner(n_arms=n_prices, gamma=0.01)
    exp3_learner_mid = EXP3_Learner(n_arms=n_prices, gamma=0.45)
    exp3_learner_high = EXP3_Learner(n_arms=n_prices, gamma=0.85)
    for t in range(0, T):
        n = int(env.draw_n(optimal_bid_phase1, 1))
        cc = env.draw_cc(optimal_bid_phase1, 1)
        pulled_arm = swucb_learner.pull_arm(margins)
        reward = [0, 0, 0]  # successes, failures, reward
        reward[0] += env.round2(pulled_arm, n, False)
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm] - cc
        swucb_learner.update(pulled_arm, reward)

        pulled_arm = cusum_learner.pull_arm(margins)
        # k = []
        # k = np.array(k)
        reward = [0, 0, 0]  # success, failures, reward, all results
        reward[0] += env.round2(pulled_arm, n, False)
        # np.append(reward[3], env.round(pulled_arm))
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm] - cc
        cusum_learner.update(pulled_arm, reward)

        # exp3 with low gamma
        pulled_arm = exp3_learner_low.pull_arm()  # normalizzare reward
        reward = [0, 0, 0]  # success, failures, reward
        reward[0] += env.round2(pulled_arm, n, False)
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm] - cc
        exp3_learner_low.update(pulled_arm, reward)

        # exp3 with mid gamma
        pulled_arm = exp3_learner_mid.pull_arm()  # normalizzare reward
        reward = [0, 0, 0]  # success, failures, reward
        reward[0] += env.round2(pulled_arm, n, False)
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm] - cc
        exp3_learner_mid.update(pulled_arm, reward)

        # exp3 with high gamma
        pulled_arm = exp3_learner_high.pull_arm()  # normalizzare reward
        reward = [0, 0, 0]  # success, failures, reward
        reward[0] += env.round2(pulled_arm, n, True)
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm] - cc
        exp3_learner_high.update(pulled_arm, reward)

    swucb_rewards_per_experiments.append(swucb_learner.collected_rewards)
    cusum_rewards_per_experiments.append(cusum_learner.collected_rewards)
    exp3low_rewards_per_experiments.append(exp3_learner_low.collected_rewards)
    exp3mid_rewards_per_experiments.append(exp3_learner_mid.collected_rewards)
    exp3high_rewards_per_experiments.append(exp3_learner_high.collected_rewards)

exp3low_reward = np.array(exp3low_rewards_per_experiments)
exp3high_reward = np.array(exp3high_rewards_per_experiments)

opt = np.ones([T])
opt[:int(T / 3)] = opt[:int(T / 3)] * opt_phase1
opt[int(T / 3):2 * int(T / 3)] = opt[int(T / 3):2 * int(T / 3)] * opt_phase2
opt[2 * int(T / 3):] = opt[2 * int(T / 3):] * opt_phase3

swucb_reward = np.array(swucb_rewards_per_experiments)
swucb_regret = np.array(opt - swucb_reward)
swucb_cum_reward = np.cumsum(swucb_reward, axis=1)
swucb_cum_regret = np.cumsum(swucb_regret, axis=1)

cusum_reward = np.array(cusum_rewards_per_experiments)
cusum_regret = np.array(opt - cusum_reward)
cusum_cum_reward = np.cumsum(cusum_reward, axis=1)
cusum_cum_regret = np.cumsum(cusum_regret, axis=1)

exp3mid_reward = np.array(exp3mid_rewards_per_experiments)
exp3mid_regret = np.array(opt - exp3mid_reward)
exp3mid_cum_reward = np.cumsum(exp3mid_reward, axis=1)
exp3mid_cum_regret = np.cumsum(exp3mid_regret, axis=1)

fig, axs = plt.subplots(2, 2, figsize=(14, 7))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Cumulative reward")
axs[0][0].plot(np.mean(swucb_cum_reward, axis=0), 'r')
axs[0][0].plot(np.mean(cusum_cum_reward, axis=0), 'm')
axs[0][0].plot(np.mean(exp3mid_cum_reward, axis=0), 'c')
axs[0][0].fill_between(range(T), np.mean(swucb_cum_reward, axis=0) - np.std(
    swucb_cum_reward, axis=0), np.mean(swucb_cum_reward, axis=0) + np.std(
    swucb_cum_reward, axis=0), color='r', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(cusum_cum_reward, axis=0) - np.std(
    cusum_cum_reward, axis=0), np.mean(cusum_cum_reward, axis=0) + np.std(
    cusum_cum_reward, axis=0), color='m', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(exp3mid_cum_reward, axis=0) - np.std(
    exp3mid_cum_reward, axis=0), np.mean(exp3mid_cum_reward, axis=0) + np.std(
    exp3mid_cum_reward, axis=0), color='c', alpha=0.2)

axs[0][0].legend(["Cumulative Reward SWUCB", "Cumulative Reward CUSUM", "Cumulative Reward EXP3"])
axs[0][0].set_title("Cumulative Reward SWUCB vs CUSUM vs EXP3")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Instantaneous Reward")
axs[0][1].plot(np.mean(swucb_reward, axis=0), 'r')
axs[0][1].plot(np.mean(cusum_reward, axis=0), 'm')
axs[0][1].plot(np.mean(exp3mid_reward, axis=0), 'c')
axs[0][1].fill_between(range(T), np.mean(swucb_reward, axis=0) - np.std(swucb_reward, axis=0),
                       np.mean(swucb_reward, axis=0) + np.std(swucb_reward, axis=0), color='r', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusum_reward, axis=0) - np.std(cusum_reward, axis=0),
                       np.mean(cusum_reward, axis=0) + np.std(cusum_reward, axis=0), color='m', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(exp3mid_reward, axis=0) - np.std(exp3mid_reward, axis=0),
                       np.mean(exp3mid_reward, axis=0) + np.std(exp3mid_reward, axis=0), color='c', alpha=0.2)
axs[0][1].legend(["Reward SWUCB", "Reward CUSUM", "Reward EXP3"])
axs[0][1].set_title("Instantaneous Reward SWUCB vs CUSUM vs EXP3")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Cumulative regret")
axs[1][0].plot(np.mean(swucb_cum_regret, axis=0), 'g')
axs[1][0].plot(np.mean(cusum_cum_regret, axis=0), 'y')
axs[1][0].plot(np.mean(exp3mid_cum_regret, axis=0), 'b')
axs[1][0].fill_between(range(T), np.mean(swucb_cum_regret, axis=0) - np.std(
    swucb_cum_regret, axis=0), np.mean(swucb_cum_regret, axis=0) + np.std(swucb_cum_regret, axis=0),
                       color='g', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusum_cum_regret, axis=0) - np.std(
    cusum_cum_regret, axis=0), np.mean(cusum_cum_regret, axis=0) + np.std(cusum_cum_regret, axis=0),
                       color='y', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(exp3mid_cum_regret, axis=0) - np.std(
    exp3mid_cum_regret, axis=0), np.mean(exp3mid_cum_regret, axis=0) + np.std(exp3mid_cum_regret, axis=0),
                       color='b', alpha=0.2)

axs[1][0].legend(["Cumulative Regret SWUCB", "Cumulative Regret CUSUM", "Cumulative Regret EXP3"])
axs[1][0].set_title("Cumulative Regret SWUCB vs CUSUM vs EXP3")

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Instantaneous regret")
axs[1][1].plot(np.mean(swucb_regret, axis=0), 'g')
axs[1][1].plot(np.mean(cusum_regret, axis=0), 'y')
axs[1][1].plot(np.mean(exp3mid_regret, axis=0), 'b')
axs[1][1].fill_between(range(T), np.mean(swucb_regret, axis=0) - np.std(
    swucb_regret, axis=0), np.mean(swucb_regret, axis=0) + np.std(swucb_regret, axis=0), color='g', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusum_regret, axis=0) - np.std(
    cusum_regret, axis=0), np.mean(cusum_regret, axis=0) + np.std(cusum_regret, axis=0), color='y', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(exp3mid_regret, axis=0) - np.std(
    exp3mid_regret, axis=0), np.mean(exp3mid_regret, axis=0) + np.std(exp3mid_regret, axis=0), color='b', alpha=0.2)
axs[1][1].legend(["Regret SWUCB", "Regret CUSUM", "Regret EXP3"])
axs[1][1].set_title("Instantaneous Regret SWUCB vs CUSUM vs EXP3")

plt.show()
