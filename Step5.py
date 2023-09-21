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
                                      #C1   C2   C3
conversion_rate_phase1 =  np.array([[0.49,0.29,0.38], #1*price
                                    [0.42,0.23,0.31], #2*price
                                    [0.35,0.18,0.24], #3*price
                                    [0.23,0.12,0.17], #4*price
                                    [0.11,0.07,0.09]  #5*price
                                    ]) 
                                      #C1   C2   C3
conversion_rate_phase2 =  np.array([[0.33, 0.18, 0.25],  # 1*price
                                    [0.25, 0.15, 0.20],  # 2*price
                                    [0.17, 0.11, 0.15],  # 3*price
                                    [0.11, 0.06, 0.10],  # 4*price
                                    [0.02, 0.03, 0.05]   # 5*price
                                    ])

                                      #C1   C2   C3
conversion_rate_phase3 =  np.array([[0.8, 0.48, 0.70],  # 1*price
                                    [0.6, 0.37, 0.56],  # 2*price
                                    [0.4, 0.28, 0.42],  # 3*price
                                    [0.2, 0.14, 0.28],  # 4*price
                                    [0.1, 0.05, 0.14]   # 5*price
                                    ]) 

earnings_phase1 = np.zeros([5,3]) # conv_rate * margin
earnings_phase2 = np.zeros([5,3]) # conv_rate * margin
earnings_phase3 = np.zeros([5,3]) # conv_rate * margin

for row in range(5):
  earnings_phase1[row,:] = conversion_rate_phase1[row,:] * margins[row]
  earnings_phase2[row,:] = conversion_rate_phase2[row,:] * margins[row]
  earnings_phase3[row,:] = conversion_rate_phase3[row,:] * margins[row]

normEarnings_phase1 = earnings_phase1.copy()
normEarnings_phase1 = normEarnings_phase1 - np.min(normEarnings_phase1)
normEarnings_phase1 = normEarnings_phase1 / np.max(normEarnings_phase1)

normEarnings_phase2 = earnings_phase2.copy()
normEarnings_phase2 = normEarnings_phase2 - np.min(normEarnings_phase2)
normEarnings_phase2 = normEarnings_phase2 / np.max(normEarnings_phase2)

normEarnings_phase3 = earnings_phase3.copy()
normEarnings_phase3 = normEarnings_phase3 - np.min(normEarnings_phase3)
normEarnings_phase3 = normEarnings_phase3 / np.max(normEarnings_phase3)


env_array = []
T = 365
for c in classes:
  env_array.append(Non_Stationary_Environment(n_prices, np.array([conversion_rate_phase1[:,c], conversion_rate_phase2[:,c], conversion_rate_phase3[:,c]]), c, T, 0))

#EXPERIMENT BEGIN


n_experiments = 100

M = 100 #number of steps to obtain reference point in change detection (for CUSUM)
eps = 0.1 #epsilon for deviation from reference point in change detection (for CUSUM)
h = np.log(T*6)**2 #threshold for change detection (for CUSUM)

ucb1_rewards_per_experiments = []
swucb_rewards_per_experiments = []
cusum_rewards_per_experiments = []

optimal1 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase1,env_array)
opt_index_phase1 = optimal1[0][0]
opt_phase1 = optimal1[2][0]
optimal_bid_index_phase1 = optimal1[1][0]
optimal_bid_phase1 = bids[int(optimal_bid_index_phase1)] #we consider the same bid (?)

optimal2 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase2,env_array)
opt_index_phase2 = optimal2[0][0]
opt_phase2 = optimal2[2][0]
optimal_bid_index_phase2 = optimal2[1][0]
optimal_bid_phase2 = bids[int(optimal_bid_index_phase2)]

optimal3 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase3,env_array)
opt_index_phase3 = optimal3[0][0]
opt_phase3 = optimal3[2][0]
optimal_bid_index_phase3 = optimal3[1][0]
optimal_bid_phase3 = bids[int(optimal_bid_index_phase3)]


for e in tqdm(range(n_experiments)):
  env = deepcopy(env_array[0])

  swucb_learner = SWUCB_Learner(n_arms = n_prices, window_size = int(T/3))
  cusum_learner = CUSUM_UCB_Learner(n_arms = n_prices, M = M, eps = eps, h = h)
  ucb1_learner = UCB1_Learner(n_arms = n_prices)
  for t in range(0, T):

    n = 0
    cc = 0

    if (env.current_phase == 0):
      n = int(env.draw_n(optimal_bid_phase1, 1))
      cc = env.draw_cc(optimal_bid_phase1, 1)
    elif (env.current_phase == 1):
      n = int(env.draw_n(optimal_bid_phase2, 1))
      cc = env.draw_cc(optimal_bid_phase2, 1)
    else:
      n = int(env.draw_n(optimal_bid_phase3, 1))
      cc = env.draw_cc(optimal_bid_phase3, 1)

    reward = [0, 0, 0]    # successes, failures, reward
    pulled_arm = swucb_learner.pull_arm(margins)
    for user in range(int(n)):
      reward[0] += env.round(pulled_arm)
    reward[1] = n - reward[0]
    reward[2] = reward[0] * margins[pulled_arm] - cc
    swucb_learner.update(pulled_arm, reward)

    pulled_arm = cusum_learner.pull_arm(margins)
    reward = [0, 0, 0]    # success, failures, reward, all results
    for user in range(int(n)):
      reward[0] += env.round(pulled_arm)
    reward[1] = n - reward[0]
    reward[2] = reward[0] * margins[pulled_arm] - cc
    cusum_learner.update(pulled_arm, reward)

    pulled_arm = ucb1_learner.pull_arm(margins)
    reward = [0, 0, 0]    # success, failures, reward
    for user in range(int(n)):
      reward[0] += env.round(pulled_arm)
    reward[1] = n - reward[0]
    reward[2] = reward[0] * margins[pulled_arm] - cc
    ucb1_learner.update(pulled_arm, reward)

  swucb_rewards_per_experiments.append(swucb_learner.collected_rewards)
  cusum_rewards_per_experiments.append(cusum_learner.collected_rewards)
  ucb1_rewards_per_experiments.append(ucb1_learner.collected_rewards)

swucb_rewards_per_experiments = np.array(swucb_rewards_per_experiments)
cusum_rewards_per_experiments = np.array(cusum_rewards_per_experiments)
ucb1_rewards_per_experiments = np.array(ucb1_rewards_per_experiments)

fig, axs = plt.subplots(2,2,figsize=(14,7))

"""opt_phase1 = opt_phase1 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
opt_phase2 = opt_phase2 * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
opt_phase3 = opt_phase3 * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)"""
opt = np.ones([T]) 
opt[:int(T/3)] = opt[:int(T/3)] * opt_phase1 
opt[int(T/3):2*int(T/3)] = opt[int(T/3):2*int(T/3)]* opt_phase2 
opt[2*int(T/3):] = opt[2*int(T/3):] * opt_phase3

"""swucb_rewards_per_experiments[:int(T/3)] = swucb_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
swucb_rewards_per_experiments[int(T/3):2*int(T/3)] = swucb_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
swucb_rewards_per_experiments[2*int(T/3):] = swucb_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3) """

"""cusum_rewards_per_experiments[:int(T/3)] = cusum_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
cusum_rewards_per_experiments[int(T/3):2*int(T/3)] = cusum_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
cusum_rewards_per_experiments[2*int(T/3):] = cusum_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)"""

"""ucb1_rewards_per_experiments[:int(T/3)] = ucb1_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
ucb1_rewards_per_experiments[int(T/3):2*int(T/3)] = ucb1_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
ucb1_rewards_per_experiments[2*int(T/3):] = ucb1_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)"""


axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(swucb_rewards_per_experiments, axis = 0)), 'tab:blue')
axs[0][0].plot(np.cumsum(np.mean(cusum_rewards_per_experiments, axis = 0)), 'tab:cyan')
axs[0][0].plot(np.cumsum(np.mean(ucb1_rewards_per_experiments, axis = 0)), 'tab:red')

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(swucb_rewards_per_experiments, axis = 0)), 'tab:orange')
axs[0][0].plot(np.cumsum(np.std(cusum_rewards_per_experiments, axis = 0)), 'tab:purple')
axs[0][0].plot(np.cumsum(np.std(ucb1_rewards_per_experiments, axis = 0)), 'tab:green')

axs[0][0].plot(np.cumsum(np.mean(opt - swucb_rewards_per_experiments, axis = 0)), 'tab:olive')
axs[0][0].plot(np.cumsum(np.mean(opt - cusum_rewards_per_experiments, axis = 0)), 'tab:pink')
axs[0][0].plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiments, axis = 0)), 'tab:brown')

axs[0][0].legend(["Reward SWUCB","Reward CUSUM","Reward UCB1","Std SWUCB","Std CUSUM","Std UCB1","Regret SWUCB","Regret CUSUM","Regret UCB1"])
axs[0][0].set_title("Cumulative SWUCB vs CUSUM vs UCB1")


axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Regret")
axs[0][1].plot(np.mean(swucb_rewards_per_experiments, axis = 0), 'r')
axs[0][1].plot(np.mean(cusum_rewards_per_experiments, axis = 0), 'm')
axs[0][1].plot(np.mean(ucb1_rewards_per_experiments, axis = 0), 'b')
axs[0][1].legend(["Reward SWUCB", "Reward CUSUM", "Reward UCB1"])
axs[0][1].set_title("Instantaneous Reward SWUCB vs CUSUM vs UCB1")

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[1][0].plot(np.std(swucb_rewards_per_experiments, axis = 0), 'b')   
axs[1][0].plot(np.std(cusum_rewards_per_experiments, axis = 0), 'c')
axs[1][0].plot(np.std(ucb1_rewards_per_experiments, axis = 0), 'r')
axs[1][0].legend(["Std SWUCB","Std CUSUM","Std UCB1"])
axs[1][0].set_title("Instantaneous Std SWUCB vs CUSUM vs UCB1")

axs[1][1].plot(np.mean(opt - swucb_rewards_per_experiments, axis = 0), 'g')
axs[1][1].plot(np.mean(opt - cusum_rewards_per_experiments, axis = 0), 'y')
axs[1][1].plot(np.mean(opt - ucb1_rewards_per_experiments, axis = 0), 'k')
axs[1][1].legend(["Regret SWUCB","Regret CUSUM","Regret UCB1"])
axs[1][1].set_title("Instantaneous Regret SWUCB vs CUSUM vs UCB1")

plt.subplots_adjust(hspace=0.33)
plt.show()

