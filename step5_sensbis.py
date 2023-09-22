from Classes.learners import UCB1_Learner, SWUCB_Learner, CUSUM_UCB_Learner
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
n_experiments = 1000

M = [10, 50, 100]  # number of steps to obtain reference point in change detection (for CUSUM)
eps = [0.5*np.log(T) / T, np.log(T) / T, 3*np.log(T) / T]   # epsilon for deviation from reference point in change detection (for CUSUM)
h = [np.log(T), np.log(T) * 2, np.log(T) * 4]  # threshold for change detection (for CUSUM)
sw = [(int(math.sqrt(T)) * 2), (int(math.sqrt(T)) * 7), (int(math.sqrt(T)) * 11)]

swucblow_rewards_per_experiments = []
swucbmid_rewards_per_experiments = []
swucbhigh_rewards_per_experiments = []

cusumMlow_rewards_per_experiments = []
cusumepslow_rewards_per_experiments = []
cusumhlow_rewards_per_experiments = []
cusumMmid_rewards_per_experiments = []
cusumepsmid_rewards_per_experiments = []
cusumhmid_rewards_per_experiments = []
cusumMhigh_rewards_per_experiments = []
cusumepshigh_rewards_per_experiments = []
cusumhhigh_rewards_per_experiments = []

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

for e in tqdm(range(n_experiments)):
    env = deepcopy(env_array[0])

    #creation of learners
    swucb_learners = []
    cusum_learners_M = []
    cusum_learners_eps = []
    cusum_learners_h = []

    for i in range(3):
        swucb_learners.append(SWUCB_Learner(n_arms=n_prices, window_size=sw[i])) 
        cusum_learners_M.append(CUSUM_UCB_Learner(n_arms=n_prices, M=M[i], eps=eps[1], h=h[1]))
        cusum_learners_eps.append(CUSUM_UCB_Learner(n_arms=n_prices, M=M[1], eps=eps[i], h=h[1]))
        cusum_learners_h.append(CUSUM_UCB_Learner(n_arms=n_prices, M=M[1], eps=eps[1], h=h[i]))
    cusum_learners = []
    cusum_learners.append(cusum_learners_M)
    cusum_learners.append(cusum_learners_eps)
    cusum_learners.append(cusum_learners_h)
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

        for swucb_learner in (swucb_learners):
            pulled_arm = swucb_learner.pull_arm(margins)
            reward = [0, 0, 0]  # successes, failures, reward
            reward[0] += env.round2(pulled_arm, n, False)
            reward[1] = n - reward[0]
            reward[2] = reward[0] * margins[pulled_arm] - cc
            swucb_learner.update(pulled_arm, reward)

        for i in range(3):
            for cusum_learner in (cusum_learners[i]):
                pulled_arm = cusum_learner.pull_arm(margins)
                reward = [0, 0, 0]  # success, failures, reward, all results
                reward[0] += env.round2(pulled_arm, n, False)
                reward[1] = n - reward[0]
                reward[2] = reward[0] * margins[pulled_arm] - cc
                cusum_learner.update(pulled_arm, reward)

        env.round2(0,0,True)
        #fittizio per aggiornare
        #update round alla fine!

    #update rewards
    swucblow_rewards_per_experiments.append(swucb_learners[0].collected_rewards)
    swucbmid_rewards_per_experiments.append(swucb_learners[1].collected_rewards)
    swucbhigh_rewards_per_experiments.append(swucb_learners[2].collected_rewards)
    cusumMlow_rewards_per_experiments.append(cusum_learners[0][0].collected_rewards)
    cusumepslow_rewards_per_experiments.append(cusum_learners[1][0].collected_rewards)
    cusumhlow_rewards_per_experiments.append(cusum_learners[2][0].collected_rewards)
    cusumMmid_rewards_per_experiments.append(cusum_learners[0][1].collected_rewards)
    cusumepsmid_rewards_per_experiments.append(cusum_learners[1][1].collected_rewards)
    cusumhmid_rewards_per_experiments.append(cusum_learners[2][1].collected_rewards)
    cusumMhigh_rewards_per_experiments.append(cusum_learners[0][2].collected_rewards)
    cusumepshigh_rewards_per_experiments.append(cusum_learners[1][2].collected_rewards)
    cusumhhigh_rewards_per_experiments.append(cusum_learners[2][2].collected_rewards)

opt = np.ones([T])
opt[:int(T / 3)] = opt[:int(T / 3)] * opt_phase1
opt[int(T / 3):2 * int(T / 3)] = opt[int(T / 3):2 * int(T / 3)] * opt_phase2
opt[2 * int(T / 3):] = opt[2 * int(T / 3):] * opt_phase3

#sliding windows analysis
swucblow_reward = np.array(swucblow_rewards_per_experiments)
swucblow_regret = np.array(opt - swucblow_reward)
swucblow_cum_reward = np.cumsum(swucblow_reward, axis=1)
swucblow_cum_regret = np.cumsum(swucblow_regret, axis=1)

swucbmid_reward = np.array(swucbmid_rewards_per_experiments)
swucbmid_regret = np.array(opt - swucbmid_reward)
swucbmid_cum_reward = np.cumsum(swucbmid_reward, axis=1)
swucbmid_cum_regret = np.cumsum(swucbmid_regret, axis=1)

swucbhigh_reward = np.array(swucbhigh_rewards_per_experiments)
swucbhigh_regret = np.array(opt - swucbhigh_reward)
swucbhigh_cum_reward = np.cumsum(swucbhigh_reward, axis=1)
swucbhigh_cum_regret = np.cumsum(swucbhigh_regret, axis=1)

#cusum analysis M
cusumMlow_reward = np.array(cusumMlow_rewards_per_experiments)
cusumMlow_regret = np.array(opt - cusumMlow_reward)
cusumMlow_cum_reward = np.cumsum(cusumMlow_reward, axis=1)
cusumMlow_cum_regret = np.cumsum(cusumMlow_regret, axis=1)

cusumMmid_reward = np.array(cusumMmid_rewards_per_experiments)
cusumMmid_regret = np.array(opt - cusumMmid_reward)
cusumMmid_cum_reward = np.cumsum(cusumMmid_reward, axis=1)
cusumMmid_cum_regret = np.cumsum(cusumMmid_regret, axis=1)

cusumMhigh_reward = np.array(cusumMhigh_rewards_per_experiments)
cusumMhigh_regret = np.array(opt - cusumMhigh_reward)
cusumMhigh_cum_reward = np.cumsum(cusumMhigh_reward, axis=1)
cusumMhigh_cum_regret = np.cumsum(cusumMhigh_regret, axis=1)

#cusum analysis epsilon
cusumepslow_reward = np.array(cusumepslow_rewards_per_experiments)
cusumepslow_regret = np.array(opt - cusumepslow_reward)
cusumepslow_cum_reward = np.cumsum(cusumepslow_reward, axis=1)
cusumepslow_cum_regret = np.cumsum(cusumepslow_regret, axis=1)

cusumepsmid_reward = np.array(cusumepsmid_rewards_per_experiments)
cusumepsmid_regret = np.array(opt - cusumepsmid_reward)
cusumepsmid_cum_reward = np.cumsum(cusumepsmid_reward, axis=1)
cusumepsmid_cum_regret = np.cumsum(cusumepsmid_regret, axis=1)

cusumepshigh_reward = np.array(cusumepshigh_rewards_per_experiments)
cusumepshigh_regret = np.array(opt - cusumepshigh_reward)
cusumepshigh_cum_reward = np.cumsum(cusumepshigh_reward, axis=1)
cusumepshigh_cum_regret = np.cumsum(cusumepshigh_regret, axis=1)

#cusum analysis h
cusumhlow_reward = np.array(cusumhlow_rewards_per_experiments)
cusumhlow_regret = np.array(opt - cusumhlow_reward)
cusumhlow_cum_reward = np.cumsum(cusumhlow_reward, axis=1)
cusumhlow_cum_regret = np.cumsum(cusumhlow_regret, axis=1)

cusumhmid_reward = np.array(cusumhmid_rewards_per_experiments)
cusumhmid_regret = np.array(opt - cusumhmid_reward)
cusumhmid_cum_reward = np.cumsum(cusumhmid_reward, axis=1)
cusumhmid_cum_regret = np.cumsum(cusumhmid_regret, axis=1)

cusumhhigh_reward = np.array(cusumhhigh_rewards_per_experiments)
cusumhhigh_regret = np.array(opt - cusumhhigh_reward)
cusumhhigh_cum_reward = np.cumsum(cusumhhigh_reward, axis=1)
cusumhhigh_cum_regret = np.cumsum(cusumhhigh_regret, axis=1)

fig, axs = plt.subplots(2, 2, figsize=(14, 7))

#######################################################################################
#regret cumulativo
#swucb
axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Cumulative regret")
axs[0][0].plot(np.mean(swucblow_cum_regret, axis=0), 'r')
axs[0][0].plot(np.mean(swucbmid_cum_regret, axis=0), 'm')
axs[0][0].plot(np.mean(swucbhigh_cum_regret, axis=0), 'c')
axs[0][0].fill_between(range(T), np.mean(swucblow_cum_regret, axis=0) - np.std(
    swucblow_cum_regret, axis=0), np.mean(swucblow_cum_regret, axis=0) + np.std(
    swucblow_cum_regret, axis=0), color='r', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(swucbmid_cum_regret, axis=0) - np.std(
    swucbmid_cum_regret, axis=0), np.mean(swucbmid_cum_regret, axis=0) + np.std(
    swucbmid_cum_regret, axis=0), color='m', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(swucbhigh_cum_regret, axis=0) - np.std(
    swucbhigh_cum_regret, axis=0), np.mean(swucbhigh_cum_regret, axis=0) + np.std(
    swucbhigh_cum_regret, axis=0), color='c', alpha=0.2)

axs[0][0].legend(["sw = 38", "sw = 133", "sw = 209"])
axs[0][0].set_title("Sensitivity analysis for SWUCB (cumulative regret)")

#cusum M
axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Cumulative regret")
axs[0][1].plot(np.mean(cusumMlow_cum_regret, axis=0), 'r')
axs[0][1].plot(np.mean(cusumMmid_cum_regret, axis=0), 'm')
axs[0][1].plot(np.mean(cusumMhigh_cum_regret, axis=0), 'c')
axs[0][1].fill_between(range(T), np.mean(cusumMlow_cum_regret, axis=0) - np.std(
    cusumMlow_cum_regret, axis=0), np.mean(cusumMlow_cum_regret, axis=0) + np.std(
    cusumMlow_cum_regret, axis=0), color='r', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusumMmid_cum_regret, axis=0) - np.std(
    cusumMmid_cum_regret, axis=0), np.mean(cusumMmid_cum_regret, axis=0) + np.std(
    cusumMmid_cum_regret, axis=0), color='m', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusumMhigh_cum_regret, axis=0) - np.std(
    cusumMhigh_cum_regret, axis=0), np.mean(cusumMhigh_cum_regret, axis=0) + np.std(
    cusumMhigh_cum_regret, axis=0), color='c', alpha=0.2)

axs[0][1].legend(["M = 10", "M = 50", "M = 100"])
axs[0][1].set_title("Sensitivity analysis for M in CUSUM-UCB (cumulative regret)")

#cusum epsilon
axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Cumulative regret")
axs[1][0].plot(np.mean(cusumepslow_cum_regret, axis=0), 'r')
axs[1][0].plot(np.mean(cusumepsmid_cum_regret, axis=0), 'm')
axs[1][0].plot(np.mean(cusumepshigh_cum_regret, axis=0), 'c')
axs[1][0].fill_between(range(T), np.mean(cusumepslow_cum_regret, axis=0) - np.std(
    cusumepslow_cum_regret, axis=0), np.mean(cusumepslow_cum_regret, axis=0) + np.std(
    cusumepslow_cum_regret, axis=0), color='r', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusumepsmid_cum_regret, axis=0) - np.std(
    cusumepsmid_cum_regret, axis=0), np.mean(cusumepsmid_cum_regret, axis=0) + np.std(
    cusumepsmid_cum_regret, axis=0), color='m', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusumepshigh_cum_regret, axis=0) - np.std(
    cusumepshigh_cum_regret, axis=0), np.mean(cusumepshigh_cum_regret, axis=0) + np.std(
    cusumepshigh_cum_regret, axis=0), color='c', alpha=0.2)

axs[1][0].legend(["eps = 0,0035", "eps = 0,007", "eps = 0,021"])
axs[1][0].set_title("Sensitivity analysis for epsilon in CUSUM-UCB (cumulative regret)")

#cusum h
axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Cumulative regret")
axs[1][1].plot(np.mean(cusumhlow_cum_regret, axis=0), 'r')
axs[1][1].plot(np.mean(cusumhmid_cum_regret, axis=0), 'm')
axs[1][1].plot(np.mean(cusumhhigh_cum_regret, axis=0), 'c')
axs[1][1].fill_between(range(T), np.mean(cusumhlow_cum_regret, axis=0) - np.std(
    cusumhlow_cum_regret, axis=0), np.mean(cusumhlow_cum_regret, axis=0) + np.std(
    cusumhlow_cum_regret, axis=0), color='r', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusumhmid_cum_regret, axis=0) - np.std(
    cusumhmid_cum_regret, axis=0), np.mean(cusumhmid_cum_regret, axis=0) + np.std(
    cusumhmid_cum_regret, axis=0), color='m', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusumhhigh_cum_regret, axis=0) - np.std(
    cusumhhigh_cum_regret, axis=0), np.mean(cusumhhigh_cum_regret, axis=0) + np.std(
    cusumhhigh_cum_regret, axis=0), color='c', alpha=0.2)

axs[1][1].legend(["h=2.56", "h=5.12", "h=10.25"])
axs[1][1].set_title("Sensitivity analysis for the threshold in CUSUM-UCB (cumulative regret)")

plt.savefig("sens_cum_rew.png")

###########################################################################################
fig, axs = plt.subplots(2, 2, figsize=(14, 7))
#regret istantaneo
#swucb
axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Instantaneous regret")
axs[0][0].plot(np.mean(swucblow_regret, axis=0), 'r')
axs[0][0].plot(np.mean(swucbmid_regret, axis=0), 'm')
axs[0][0].plot(np.mean(swucbhigh_regret, axis=0), 'c')
axs[0][0].fill_between(range(T), np.mean(swucblow_regret, axis=0) - np.std(
    swucblow_regret, axis=0), np.mean(swucblow_regret, axis=0) + np.std(
    swucblow_regret, axis=0), color='r', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(swucbmid_regret, axis=0) - np.std(
    swucbmid_regret, axis=0), np.mean(swucbmid_regret, axis=0) + np.std(
    swucbmid_regret, axis=0), color='m', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(swucbhigh_regret, axis=0) - np.std(
    swucbhigh_regret, axis=0), np.mean(swucbhigh_regret, axis=0) + np.std(
    swucbhigh_regret, axis=0), color='c', alpha=0.2)

axs[0][0].legend(["sw = 38", "sw = 133", "sw = 209"])
axs[0][0].set_title("Sensitivity analysis for SWUCB (Instantaneous regret)")

#cusum M
axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Instantaneous regret")
axs[0][1].plot(np.mean(cusumMlow_regret, axis=0), 'r')
axs[0][1].plot(np.mean(cusumMmid_regret, axis=0), 'm')
axs[0][1].plot(np.mean(cusumMhigh_regret, axis=0), 'c')
axs[0][1].fill_between(range(T), np.mean(cusumMlow_regret, axis=0) - np.std(
    cusumMlow_regret, axis=0), np.mean(cusumMlow_regret, axis=0) + np.std(
    cusumMlow_regret, axis=0), color='r', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusumMmid_regret, axis=0) - np.std(
    cusumMmid_regret, axis=0), np.mean(cusumMmid_regret, axis=0) + np.std(
    cusumMmid_regret, axis=0), color='m', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusumMhigh_regret, axis=0) - np.std(
    cusumMhigh_regret, axis=0), np.mean(cusumMhigh_regret, axis=0) + np.std(
    cusumMhigh_regret, axis=0), color='c', alpha=0.2)

axs[0][1].legend(["M = 10", "M = 50", "M = 100"])
axs[0][1].set_title("Sensitivity analysis for M in CUSUM-UCB (Instantaneous regret)")

#cusum epsilon
axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Instantaneous regret")
axs[1][0].plot(np.mean(cusumepslow_regret, axis=0), 'r')
axs[1][0].plot(np.mean(cusumepsmid_regret, axis=0), 'm')
axs[1][0].plot(np.mean(cusumepshigh_regret, axis=0), 'c')
axs[1][0].fill_between(range(T), np.mean(cusumepslow_regret, axis=0) - np.std(
    cusumepslow_regret, axis=0), np.mean(cusumepslow_regret, axis=0) + np.std(
    cusumepslow_regret, axis=0), color='r', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusumepsmid_regret, axis=0) - np.std(
    cusumepsmid_regret, axis=0), np.mean(cusumepsmid_regret, axis=0) + np.std(
    cusumepsmid_regret, axis=0), color='m', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusumepshigh_regret, axis=0) - np.std(
    cusumepshigh_regret, axis=0), np.mean(cusumepshigh_regret, axis=0) + np.std(
    cusumepshigh_regret, axis=0), color='c', alpha=0.2)

axs[1][0].legend(["eps = 0,0035", "eps = 0,007", "eps = 0,021"])
axs[1][0].set_title("Sensitivity analysis for epsilon in CUSUM-UCB (Instantaneous regret)")

#cusum h
axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Instantaneous regret")
axs[1][1].plot(np.mean(cusumhlow_regret, axis=0), 'r')
axs[1][1].plot(np.mean(cusumhmid_regret, axis=0), 'm')
axs[1][1].plot(np.mean(cusumhhigh_regret, axis=0), 'c')
axs[1][1].fill_between(range(T), np.mean(cusumhlow_regret, axis=0) - np.std(
    cusumhlow_regret, axis=0), np.mean(cusumhlow_regret, axis=0) + np.std(
    cusumhlow_regret, axis=0), color='r', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusumhmid_regret, axis=0) - np.std(
    cusumhmid_regret, axis=0), np.mean(cusumhmid_regret, axis=0) + np.std(
    cusumhmid_regret, axis=0), color='m', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusumhhigh_regret, axis=0) - np.std(
    cusumhhigh_regret, axis=0), np.mean(cusumhhigh_regret, axis=0) + np.std(
    cusumhhigh_regret, axis=0), color='c', alpha=0.2)

axs[1][1].legend(["h=2.56", "h=5.12", "h=10.25"])
axs[1][1].set_title("Sensitivity analysis for the threshold in CUSUM-UCB (Instantaneous regret)")

plt.savefig("sens_inst_regret.png")

###########################################################################################
fig, axs = plt.subplots(2, 2, figsize=(14, 7))
#reward istantaneo
#swucb
axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Instantaneous reward")
axs[0][0].plot(np.mean(swucblow_reward, axis=0), 'r')
axs[0][0].plot(np.mean(swucbmid_reward, axis=0), 'm')
axs[0][0].plot(np.mean(swucbhigh_reward, axis=0), 'c')
axs[0][0].fill_between(range(T), np.mean(swucblow_reward, axis=0) - np.std(
    swucblow_reward, axis=0), np.mean(swucblow_reward, axis=0) + np.std(
    swucblow_reward, axis=0), color='r', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(swucbmid_reward, axis=0) - np.std(
    swucbmid_reward, axis=0), np.mean(swucbmid_reward, axis=0) + np.std(
    swucbmid_reward, axis=0), color='m', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(swucbhigh_reward, axis=0) - np.std(
    swucbhigh_reward, axis=0), np.mean(swucbhigh_reward, axis=0) + np.std(
    swucbhigh_reward, axis=0), color='c', alpha=0.2)

axs[0][0].legend(["sw = 38", "sw = 133", "sw = 209"])
axs[0][0].set_title("Sensitivity analysis for SWUCB (Instantaneous reward)")

#cusum M
axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Instantaneous reward")
axs[0][1].plot(np.mean(cusumMlow_reward, axis=0), 'r')
axs[0][1].plot(np.mean(cusumMmid_reward, axis=0), 'm')
axs[0][1].plot(np.mean(cusumMhigh_reward, axis=0), 'c')
axs[0][1].fill_between(range(T), np.mean(cusumMlow_reward, axis=0) - np.std(
    cusumMlow_reward, axis=0), np.mean(cusumMlow_reward, axis=0) + np.std(
    cusumMlow_reward, axis=0), color='r', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusumMmid_reward, axis=0) - np.std(
    cusumMmid_reward, axis=0), np.mean(cusumMmid_reward, axis=0) + np.std(
    cusumMmid_reward, axis=0), color='m', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusumMhigh_reward, axis=0) - np.std(
    cusumMhigh_reward, axis=0), np.mean(cusumMhigh_reward, axis=0) + np.std(
    cusumMhigh_reward, axis=0), color='c', alpha=0.2)

axs[0][1].legend(["M = 10", "M = 50", "M = 100"])
axs[0][1].set_title("Sensitivity analysis for M in CUSUM-UCB (Instantaneous reward)")

#cusum epsilon
axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Instantaneous reward")
axs[1][0].plot(np.mean(cusumepslow_reward, axis=0), 'r')
axs[1][0].plot(np.mean(cusumepsmid_reward, axis=0), 'm')
axs[1][0].plot(np.mean(cusumepshigh_reward, axis=0), 'c')
axs[1][0].fill_between(range(T), np.mean(cusumepslow_reward, axis=0) - np.std(
    cusumepslow_reward, axis=0), np.mean(cusumepslow_reward, axis=0) + np.std(
    cusumepslow_reward, axis=0), color='r', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusumepsmid_reward, axis=0) - np.std(
    cusumepsmid_reward, axis=0), np.mean(cusumepsmid_reward, axis=0) + np.std(
    cusumepsmid_reward, axis=0), color='m', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusumepshigh_reward, axis=0) - np.std(
    cusumepshigh_reward, axis=0), np.mean(cusumepshigh_reward, axis=0) + np.std(
    cusumepshigh_reward, axis=0), color='c', alpha=0.2)

axs[1][0].legend(["eps = 0,0035", "eps = 0,007", "eps = 0,021"])
axs[1][0].set_title("Sensitivity analysis for epsilon in CUSUM-UCB (Instantaneous reward)")

#cusum h
axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Instantaneous reward")
axs[1][1].plot(np.mean(cusumhlow_reward, axis=0), 'r')
axs[1][1].plot(np.mean(cusumhmid_reward, axis=0), 'm')
axs[1][1].plot(np.mean(cusumhhigh_reward, axis=0), 'c')
axs[1][1].fill_between(range(T), np.mean(cusumhlow_reward, axis=0) - np.std(
    cusumhlow_reward, axis=0), np.mean(cusumhlow_reward, axis=0) + np.std(
    cusumhlow_reward, axis=0), color='r', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusumhmid_reward, axis=0) - np.std(
    cusumhmid_reward, axis=0), np.mean(cusumhmid_reward, axis=0) + np.std(
    cusumhmid_reward, axis=0), color='m', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusumhhigh_reward, axis=0) - np.std(
    cusumhhigh_reward, axis=0), np.mean(cusumhhigh_reward, axis=0) + np.std(
    cusumhhigh_reward, axis=0), color='c', alpha=0.2)

axs[1][1].legend(["h=2.56", "h=5.12", "h=10.25"])
axs[1][1].set_title("Sensitivity analysis for the threshold in CUSUM-UCB (Instantaneous reward)")

plt.savefig("sens_inst_reward.png")

###########################################################################################
fig, axs = plt.subplots(2, 2, figsize=(14, 7))
#reward cumulativo
#swucb
axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Cumulative Reward")
axs[0][0].plot(np.mean(swucblow_cum_reward, axis=0), 'r')
axs[0][0].plot(np.mean(swucbmid_cum_reward, axis=0), 'm')
axs[0][0].plot(np.mean(swucbhigh_cum_reward, axis=0), 'c')
axs[0][0].fill_between(range(T), np.mean(swucblow_cum_reward, axis=0) - np.std(
    swucblow_cum_reward, axis=0), np.mean(swucblow_cum_reward, axis=0) + np.std(
    swucblow_cum_reward, axis=0), color='r', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(swucbmid_cum_reward, axis=0) - np.std(
    swucbmid_cum_reward, axis=0), np.mean(swucbmid_cum_reward, axis=0) + np.std(
    swucbmid_cum_reward, axis=0), color='m', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(swucbhigh_cum_reward, axis=0) - np.std(
    swucbhigh_cum_reward, axis=0), np.mean(swucbhigh_cum_reward, axis=0) + np.std(
    swucbhigh_cum_reward, axis=0), color='c', alpha=0.2)

axs[0][0].legend(["sw = 38", "sw = 133", "sw = 209"])
axs[0][0].set_title("Sensitivity analysis for SWUCB (Cumulative Reward)")

#cusum M
axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Cumulative Reward")
axs[0][1].plot(np.mean(cusumMlow_cum_reward, axis=0), 'r')
axs[0][1].plot(np.mean(cusumMmid_cum_reward, axis=0), 'm')
axs[0][1].plot(np.mean(cusumMhigh_cum_reward, axis=0), 'c')
axs[0][1].fill_between(range(T), np.mean(cusumMlow_cum_reward, axis=0) - np.std(
    cusumMlow_cum_reward, axis=0), np.mean(cusumMlow_cum_reward, axis=0) + np.std(
    cusumMlow_cum_reward, axis=0), color='r', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusumMmid_cum_reward, axis=0) - np.std(
    cusumMmid_cum_reward, axis=0), np.mean(cusumMmid_cum_reward, axis=0) + np.std(
    cusumMmid_cum_reward, axis=0), color='m', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(cusumMhigh_cum_reward, axis=0) - np.std(
    cusumMhigh_cum_reward, axis=0), np.mean(cusumMhigh_cum_reward, axis=0) + np.std(
    cusumMhigh_cum_reward, axis=0), color='c', alpha=0.2)

axs[0][1].legend(["M = 10", "M = 50", "M = 100"])
axs[0][1].set_title("Sensitivity analysis for M in CUSUM-UCB (Cumulative Reward)")

#cusum epsilon
axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Cumulative Reward")
axs[1][0].plot(np.mean(cusumepslow_cum_reward, axis=0), 'r')
axs[1][0].plot(np.mean(cusumepsmid_cum_reward, axis=0), 'm')
axs[1][0].plot(np.mean(cusumepshigh_cum_reward, axis=0), 'c')
axs[1][0].fill_between(range(T), np.mean(cusumepslow_cum_reward, axis=0) - np.std(
    cusumepslow_cum_reward, axis=0), np.mean(cusumepslow_cum_reward, axis=0) + np.std(
    cusumepslow_cum_reward, axis=0), color='r', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusumepsmid_cum_reward, axis=0) - np.std(
    cusumepsmid_cum_reward, axis=0), np.mean(cusumepsmid_cum_reward, axis=0) + np.std(
    cusumepsmid_cum_reward, axis=0), color='m', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(cusumepshigh_cum_reward, axis=0) - np.std(
    cusumepshigh_cum_reward, axis=0), np.mean(cusumepshigh_cum_reward, axis=0) + np.std(
    cusumepshigh_cum_reward, axis=0), color='c', alpha=0.2)

axs[1][0].legend(["eps = 0,0035", "eps = 0,007", "eps = 0,021"])
axs[1][0].set_title("Sensitivity analysis for epsilon in CUSUM-UCB (Cumulative Reward)")

#cusum h
axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Cumulative Reward")
axs[1][1].plot(np.mean(cusumhlow_cum_reward, axis=0), 'r')
axs[1][1].plot(np.mean(cusumhmid_cum_reward, axis=0), 'm')
axs[1][1].plot(np.mean(cusumhhigh_cum_reward, axis=0), 'c')
axs[1][1].fill_between(range(T), np.mean(cusumhlow_cum_reward, axis=0) - np.std(
    cusumhlow_cum_reward, axis=0), np.mean(cusumhlow_cum_reward, axis=0) + np.std(
    cusumhlow_cum_reward, axis=0), color='r', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusumhmid_cum_reward, axis=0) - np.std(
    cusumhmid_cum_reward, axis=0), np.mean(cusumhmid_cum_reward, axis=0) + np.std(
    cusumhmid_cum_reward, axis=0), color='m', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusumhhigh_cum_reward, axis=0) - np.std(
    cusumhhigh_cum_reward, axis=0), np.mean(cusumhhigh_cum_reward, axis=0) + np.std(
    cusumhhigh_cum_reward, axis=0), color='c', alpha=0.2)

axs[1][1].legend(["h=2.56", "h=5.12", "h=10.25"])
axs[1][1].set_title("Sensitivity analysis for the threshold in CUSUM-UCB (Cumulative Reward)")
plt.savefig("cumulative_reward_sens.png")

####
"""
axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Instantaneous regret")
axs[1][1].plot(np.mean(swucb_regret, axis=0), 'g')
axs[1][1].plot(np.mean(cusum_regret, axis=0), 'y')
axs[1][1].plot(np.mean(ucb1_regret, axis=0), 'b')
axs[1][1].fill_between(range(T), np.mean(swucb_regret, axis=0) - np.std(
    swucb_regret, axis=0), np.mean(swucb_regret, axis=0) + np.std(swucb_regret, axis=0), color='g', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(cusum_regret, axis=0) - np.std(
    cusum_regret, axis=0), np.mean(cusum_regret, axis=0) + np.std(cusum_regret, axis=0), color='y', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(ucb1_regret, axis=0) - np.std(
    ucb1_regret, axis=0), np.mean(ucb1_regret, axis=0) + np.std(ucb1_regret, axis=0), color='b', alpha=0.2)
axs[1][1].legend(["Regret SWUCB", "Regret CUSUM", "Regret UCB1"])
axs[1][1].set_title("Instantaneous Regret SWUCB vs CUSUM vs UCB1")"""

plt.show()
