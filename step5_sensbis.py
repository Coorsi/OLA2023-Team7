from Classes.learners import SWUCB_Learner, CUSUM_UCB_Learner
from Classes.enviroment import Non_Stationary_Environment
from Classes.clairvoyant import clairvoyant
import math

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
# C1   C2   C3
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

window_sizes = [int(2 * math.sqrt(T)), int(3 * math.sqrt(T)), int(4 * math.sqrt(T)), int(5 * math.sqrt(T))]
M_values = [50, 100, 200]
eps_values = [0.1, 0.3, 0.5]
h_values = [0.05 * np.log(T), np.log(T) * 0.5, np.log(T) * 5]
params = [window_sizes, M_values, eps_values, h_values]

k = 3  # choose k (0=winsize, 1=M, 2=eps, 3=h)

swucb_rewards_per_experiments = [[] for i in range(len(params[k]))]
cusum_rewards_per_experiments = [[] for i in range(len(params[k]))]

optimal1 = clairvoyant(classes, bids, prices, margins, conversion_rate_phase1, env_array)
opt_index_phase1 = optimal1[0][0]
opt_phase1 = optimal1[2][0]
optimal_bid_index_phase1 = optimal1[1][0]
optimal_bid_phase1 = bids[int(optimal_bid_index_phase1)]  # we consider the same bid (?)

optimal2 = clairvoyant(classes, bids, prices, margins, conversion_rate_phase2, env_array)
opt_index_phase2 = optimal2[0][0]
opt_phase2 = optimal2[2][0]
optimal_bid_index_phase2 = optimal2[1][0]
optimal_bid_phase2 = bids[int(optimal_bid_index_phase2)]

optimal3 = clairvoyant(classes, bids, prices, margins, conversion_rate_phase3, env_array)
opt_index_phase3 = optimal3[0][0]
opt_phase3 = optimal3[2][0]
optimal_bid_index_phase3 = optimal3[1][0]
optimal_bid_phase3 = bids[int(optimal_bid_index_phase3)]

if k == 0:
    for e in tqdm(range(n_experiments)):
        env = deepcopy(env_array[0])
        swucb_learner = [SWUCB_Learner(n_arms=n_prices, window_size=params[k][i]) for i in range(len(params[k]))]
        for i in range(len(params[k])):
            env.current_phase = 0
            for t in range(0, T):
                n = 0
                cc = 0
                if env.current_phase == 0:
                    n = int(env.draw_n(optimal_bid_phase1, 1))
                    cc = env.draw_cc(optimal_bid_phase1, 1)
                elif env.current_phase == 1:
                    n = int(env.draw_n(optimal_bid_phase2, 1))
                    cc = env.draw_cc(optimal_bid_phase2, 1)
                else:
                    n = int(env.draw_n(optimal_bid_phase3, 1))
                    cc = env.draw_cc(optimal_bid_phase3, 1)

                pulled_arm = swucb_learner[i].pull_arm(margins)
                reward = [0, 0, 0]  # successes, failures, reward
                reward[0] += env.round2(pulled_arm, n, True)
                reward[1] = n - reward[0]
                reward[2] = reward[0] * margins[pulled_arm] - cc
                swucb_learner[i].update(pulled_arm, reward)

        for i in range(len(params[k])):
            swucb_rewards_per_experiments[i].append(swucb_learner[i].collected_rewards)
else:
    for e in tqdm(range(n_experiments)):
        env = deepcopy(env_array[0])
        if k == 1:
            cusum_learner = [CUSUM_UCB_Learner(n_arms=n_prices, M=params[k][i], eps=params[2][1], h=params[
                3][1]) for i in range(len(params[k]))]
        elif k == 2:
            cusum_learner = [CUSUM_UCB_Learner(n_arms=n_prices, M=params[1][1], eps=params[k][i], h=params[
                3][1]) for i in range(len(params[k]))]
        elif k == 3:
            cusum_learner = [CUSUM_UCB_Learner(n_arms=n_prices, M=params[1][1], eps=params[2][1], h=params[
                k][i]) for i in range(len(params[k]))]
        for i in range(len(params[k])):
            env.current_phase = 0
            for t in range(0, T):
                n = 0
                cc = 0
                if env.current_phase == 0:
                    n = int(env.draw_n(optimal_bid_phase1, 1))
                    cc = env.draw_cc(optimal_bid_phase1, 1)
                elif env.current_phase == 1:
                    n = int(env.draw_n(optimal_bid_phase2, 1))
                    cc = env.draw_cc(optimal_bid_phase2, 1)
                else:
                    n = int(env.draw_n(optimal_bid_phase3, 1))
                    cc = env.draw_cc(optimal_bid_phase3, 1)

                pulled_arm = cusum_learner[i].pull_arm(margins)
                reward = [0, 0, 0]  # successes, failures, reward
                reward[0] += env.round2(pulled_arm, n, True)
                reward[1] = n - reward[0]
                reward[2] = reward[0] * margins[pulled_arm] - cc
                cusum_learner[i].update(pulled_arm, reward)

        for i in range(len(params[k])):
            cusum_rewards_per_experiments[i].append(cusum_learner[i].collected_rewards)

opt = np.ones([T])
opt[:int(T / 3)] = opt[:int(T / 3)] * opt_phase1
opt[int(T / 3):2 * int(T / 3)] = opt[int(T / 3):2 * int(T / 3)] * opt_phase2
opt[2 * int(T / 3):] = opt[2 * int(T / 3):] * opt_phase3
print(opt.shape)

swucb_reward = []
swucb_regret = []
swucb_cum_reward = []
swucb_cum_regret = []
cusum_reward = []
cusum_regret = []
cusum_cum_reward = []
cusum_cum_regret = []

if k == 0:
    for i in range(len(params[k])):
        swucb_reward.append(np.array(swucb_rewards_per_experiments[i]))
        swucb_regret.append(np.array(opt - swucb_reward[i]))
        swucb_cum_reward.append(np.cumsum(swucb_reward[i], axis=1))
        swucb_cum_regret.append(np.cumsum(swucb_regret[i], axis=1))
else:
    for i in range(len(params[k])):
        cusum_reward.append(np.array(cusum_rewards_per_experiments[i]))
        cusum_regret.append(np.array(opt - cusum_reward[i]))
        cusum_cum_reward.append(np.cumsum(cusum_reward[i], axis=1))
        cusum_cum_regret.append(np.cumsum(cusum_regret[i], axis=1))

fig, axs = plt.subplots(2, 2, figsize=(14, 7))
lines = [[] for i in range(4)]

if k == 0:
    for i in range(len(params[k])):
        if i == 0:
            color = 'r'
        elif i == 1:
            color = 'm'
        elif i == 2:
            color = 'g'
        elif i == 3:
            color = 'y'
        else:
            color = 'b'
        axs[0][0].set_xlabel("t")
        axs[0][0].set_ylabel("Cumulative reward")
        axs[0][0].plot(np.mean(swucb_cum_reward[i], axis=0), color, label=str(params[k][i]))
        axs[0][0].fill_between(range(T), np.mean(swucb_cum_reward[i], axis=0) - np.std(
            swucb_cum_reward[i], axis=0), np.mean(swucb_cum_reward[i], axis=0) + np.std(
            swucb_cum_reward[i], axis=0), color=color, alpha=0.2)
        axs[0][0].set_title("Cumulative Reward")

        axs[0][1].set_xlabel("t")
        axs[0][1].set_ylabel("Instantaneous Reward")
        axs[0][1].plot(np.mean(swucb_reward[i], axis=0), color, label=str(params[k][i]))
        axs[0][1].fill_between(range(T), np.mean(swucb_reward[i], axis=0) - np.std(swucb_reward[i], axis=0), np.mean(
            swucb_reward[i], axis=0) + np.std(swucb_reward[i], axis=0), color=color, alpha=0.2)
        axs[0][1].set_title("Instantaneous Reward")

        axs[1][0].set_xlabel("t")
        axs[1][0].set_ylabel("Cumulative regret")
        axs[1][0].plot(np.mean(swucb_cum_regret[i], axis=0), color, label=str(params[k][i]))
        axs[1][0].fill_between(range(T), np.mean(swucb_cum_regret[i], axis=0) - np.std(swucb_cum_regret[i], axis=0), np.
                               mean(swucb_cum_regret[i], axis=0) + np.std(swucb_cum_regret[i], axis=0),
                               color=color, alpha=0.2)
        axs[1][0].set_title("Cumulative Regret")

        axs[1][1].set_xlabel("t")
        axs[1][1].set_ylabel("Instantaneous regret")
        axs[1][1].plot(np.mean(swucb_regret[i], axis=0), color, label=str(params[k][i]))
        axs[1][1].fill_between(range(T), np.mean(swucb_regret[i], axis=0) - np.std(swucb_regret[i], axis=0), np.mean(
            swucb_regret[i], axis=0) + np.std(swucb_regret[i], axis=0), color=color, alpha=0.2)
        axs[1][1].set_title("Instantaneous Regret")
else:
    for i in range(len(params[k])):
        if i == 0:
            color = 'r'
        elif i == 1:
            color = 'm'
        elif i == 2:
            color = 'g'
        elif i == 3:
            color = 'y'
        else:
            color = 'b'
        axs[0][0].set_xlabel("t")
        axs[0][0].set_ylabel("Cumulative reward")
        axs[0][0].plot(np.mean(cusum_cum_reward[i], axis=0), color, label=str(params[k][i]))
        axs[0][0].fill_between(range(T), np.mean(cusum_cum_reward[i], axis=0) - np.std(
            cusum_cum_reward[i], axis=0), np.mean(cusum_cum_reward[i], axis=0) + np.std(
            cusum_cum_reward[i], axis=0), color=color, alpha=0.2)
        axs[0][0].set_title("Cumulative Reward")

        axs[0][1].set_xlabel("t")
        axs[0][1].set_ylabel("Instantaneous Reward")
        axs[0][1].plot(np.mean(cusum_reward[i], axis=0), color, label=str(params[k][i]))
        axs[0][1].fill_between(range(T), np.mean(cusum_reward[i], axis=0) - np.std(cusum_reward[i], axis=0),
                               np.mean(cusum_reward[i], axis=0) + np.std(cusum_reward[i], axis=0), color=color,
                               alpha=0.2)
        axs[0][1].set_title("Instantaneous Reward")

        axs[1][0].set_xlabel("t")
        axs[1][0].set_ylabel("Cumulative regret")
        axs[1][0].plot(np.mean(cusum_cum_regret[i], axis=0), color, label=str(params[k][i]))
        axs[1][0].fill_between(range(T), np.mean(cusum_cum_regret[i], axis=0) - np.std(cusum_cum_regret[i], axis=0), np.
                               mean(cusum_cum_regret[i], axis=0) + np.std(cusum_cum_regret[i], axis=0),
                               color=color, alpha=0.2)
        axs[1][0].set_title("Cumulative Regret")

        axs[1][1].set_xlabel("t")
        axs[1][1].set_ylabel("Instantaneous regret")
        axs[1][1].plot(np.mean(cusum_regret[i], axis=0), color, label=str(params[k][i]))
        axs[1][1].fill_between(range(T), np.mean(cusum_regret[i], axis=0) - np.std(cusum_regret[i], axis=0), np.mean(
            cusum_regret[i], axis=0) + np.std(cusum_regret[i], axis=0), color=color, alpha=0.2)
        axs[1][1].set_title("Instantaneous Regret")

axs[0][0].legend()
axs[0][1].legend()
axs[1][0].legend()
axs[1][1].legend()

plt.show()
