import math

from Classes.learners import Learner,TS_Learner,UCB1_Learner,SWTS_Learner,SWUCB_Learner,CUSUM_UCB_Learner
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
prices = price*np.array([1,2,3,4,5])
margins = np.array([prices[i]-cost_of_product for i in range(n_prices)])
classes = np.array([0,1,2])
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
  env_array.append(Non_Stationary_Environment(n_prices, np.array([conversion_rate_phase1[:,c],
                                            conversion_rate_phase2[:,c], conversion_rate_phase3[:,c]]), c, T, 0))

#EXPERIMENT BEGIN


n_experiments = 100
window_sizes = [int(2*math.sqrt(T)), int(3*math.sqrt(T)), int(4*math.sqrt(T)), int(5*math.sqrt(T))]
M_values = [50, 100, 200]
eps_values = [0.1, 0.2, 0.3]
h_values = [np.log(T) * 2, np.log(T) * 3, np.log(T) * 4]

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
opt = np.ones([T]) 
opt[:int(T/3)] = opt[:int(T/3)] * opt_phase1 
opt[int(T/3):2*int(T/3)] = opt[int(T/3):2*int(T/3)]* opt_phase2 
opt[2*int(T/3):] = opt[2*int(T/3):] * opt_phase3



fig, axs = plt.subplots(2,2, figsize=(14,7))
colors  = ['tab:orange', 'tab:purple', 'tab:green']
for window_size in window_sizes:
    swucb_rewards_per_experiments = []
    for e in tqdm(range(n_experiments)):
        env_swucb = deepcopy(env_array[0])

        swucb_learner = SWUCB_Learner(n_arms = n_prices, window_size = window_size)

        for t in range(0, T):

            pulled_arm = swucb_learner.pull_arm()
            reward = env_swucb.round(pulled_arm)
            swucb_learner.update(pulled_arm, reward)

        swucb_rewards_per_experiments.append(swucb_learner.collected_rewards)


    swucb_rewards_per_experiments = np.array(swucb_rewards_per_experiments)


    swucb_rewards_per_experiments[:int(T/3)] = swucb_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
    swucb_rewards_per_experiments[int(T/3):2*int(T/3)] = swucb_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
    swucb_rewards_per_experiments[2*int(T/3):] = swucb_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)
    
    i = window_sizes.index(window_size)

    axs[0][0].plot(np.cumsum(np.mean(opt - swucb_rewards_per_experiments, axis = 0)), colors[i])

axs[0][0].set_xlabel("time")
axs[0][0].set_ylabel("Regret")

axs[0][0].legend(["Regret SWUCB(WS ="+str(window_size)+")" for window_size in window_sizes])
axs[0][0].set_title("Cumulative Regret of SWUCB with different window sizes")

for m in M_values:
    cusum_rewards_per_experiments = []
    for e in tqdm(range(n_experiments)):
        env_cusum = deepcopy(env_array[0])

        cusum_learner = CUSUM_UCB_Learner(n_arms = n_prices, M = m, eps = eps_values[0], h = h_values[0])

        for t in range(0, T):

            pulled_arm = cusum_learner.pull_arm()
            reward = env_cusum.round(pulled_arm)
            cusum_learner.update(pulled_arm, reward)

        cusum_rewards_per_experiments.append(cusum_learner.collected_rewards)

    cusum_rewards_per_experiments = np.array(cusum_rewards_per_experiments)
   
    cusum_rewards_per_experiments[:int(T/3)] = cusum_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
    cusum_rewards_per_experiments[int(T/3):2*int(T/3)] = cusum_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
    cusum_rewards_per_experiments[2*int(T/3):] = cusum_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)

    i = M_values.index(m)

    axs[0][1].plot(np.cumsum(np.mean(opt - cusum_rewards_per_experiments, axis = 0)), colors[i])

axs[0][1].set_xlabel("time")
axs[0][1].set_ylabel("Regret")

axs[0][1].legend(["Regret CUSUM(M ="+str(m)+")" for m in M_values])
axs[0][1].set_title("Cumulative Regret of CUSUM with different M values")

for eps in eps_values:

    cusum_rewards_per_experiments = []
    for e in tqdm(range(n_experiments)):
        env_cusum = deepcopy(env_array[0])

        cusum_learner = CUSUM_UCB_Learner(n_arms = n_prices, M = M_values[1], eps = eps, h = h_values[0])
        for t in range(0, T):

            pulled_arm = cusum_learner.pull_arm()
            reward = env_cusum.round(pulled_arm)
            cusum_learner.update(pulled_arm, reward)

        cusum_rewards_per_experiments.append(cusum_learner.collected_rewards)

    cusum_rewards_per_experiments = np.array(cusum_rewards_per_experiments)

    cusum_rewards_per_experiments[:int(T/3)] = cusum_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
    cusum_rewards_per_experiments[int(T/3):2*int(T/3)] = cusum_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
    cusum_rewards_per_experiments[2*int(T/3):] = cusum_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)

    i = eps_values.index(eps)
    axs[1][0].plot(np.cumsum(np.mean(opt - cusum_rewards_per_experiments, axis = 0)), colors[i])

axs[1][0].set_xlabel("time")
axs[1][0].set_ylabel("Regret")

axs[1][0].legend(["Regret CUSUM(eps ="+str(eps)+")" for eps in eps_values])
axs[1][0].set_title("Cumulative Regret of CUSUM with different epsilon values")

for h in h_values:
    cusum_rewards_per_experiments = []
    for e in tqdm(range(n_experiments)):
        env_cusum = deepcopy(env_array[0])

        cusum_learner = CUSUM_UCB_Learner(n_arms = n_prices, M = M_values[1], eps = eps_values[0], h = h)
        for t in range(0, T):

            pulled_arm = cusum_learner.pull_arm()
            reward = env_cusum.round(pulled_arm)
            cusum_learner.update(pulled_arm, reward)

        cusum_rewards_per_experiments.append(cusum_learner.collected_rewards)

    cusum_rewards_per_experiments = np.array(cusum_rewards_per_experiments)

    cusum_rewards_per_experiments[:int(T/3)] = cusum_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
    cusum_rewards_per_experiments[int(T/3):2*int(T/3)] = cusum_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
    cusum_rewards_per_experiments[2*int(T/3):] = cusum_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)

    i = h_values.index(h)
    axs[1][1].plot(np.cumsum(np.mean(opt - cusum_rewards_per_experiments, axis = 0)), colors[i])

axs[1][1].set_xlabel("time")
axs[1][1].set_ylabel("Regret")

axs[1][1].legend(["Regret CUSUM(h ="+str(h)+")" for h in h_values])
axs[1][1].set_title("Cumulative Regret of CUSUM with different h(threshold) values")

plt.subplots_adjust(hspace=0.33)
plt.show()