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
conversion_rate_phase1 =  np.array([[0.93,0.95,0.77], #1*price
                                    [0.82,0.84,0.42], #2*price
                                    [0.51,0.64,0.29], #3*price
                                    [0.38,0.50,0.21], #4*price
                                    [0.09,0.18,0.11]  #5*price
                                    ])
                                      #C1   C2   C3
conversion_rate_phase2 =  np.array([[0.95, 0.87, 0.75],  # 1*price
                                    [0.89, 0.78, 0.40],  # 2*price
                                    [0.75, 0.62, 0.27],  # 3*price
                                    [0.30, 0.48, 0.19],  # 4*price
                                    [0.15, 0.17, 0.09]   # 5*price
                                    ])

                                      #C1   C2   C3
conversion_rate_phase3 =  np.array([[0.95, 0.97, 0.80],  # 1*price
                                    [0.85, 0.90, 0.48],  # 2*price
                                    [0.50, 0.68, 0.31],  # 3*price
                                    [0.44, 0.54, 0.23],  # 4*price
                                    [0.13, 0.20, 0.12]   # 5*price
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
  env_array.append(Non_Stationary_Environment(n_prices, np.array([normEarnings_phase1[:,c], normEarnings_phase2[:,c], normEarnings_phase3[:,c]]), c, T, 0))

#EXPERIMENT BEGIN


n_experiments = 100

M = 100 #number of steps to obtain reference point in change detection (for CUSUM)
eps = 0.1 #epsilon for deviation from reference point in change detection (for CUSUM)
h = np.log(T)*2 #threshold for change detection (for CUSUM)

swucb_rewards_per_experiments = []
cusum_rewards_per_experiments = []

opt_index_phase1 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase1,env_array)[0][0])
opt_phase1 = normEarnings_phase1[opt_index_phase1][0]
optimal_bid_index_phase1 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase1,env_array)[1][0]
optimal_bid_phase1 = bids[int(optimal_bid_index_phase1)]

opt_index_phase2 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase2,env_array)[0][0])
opt_phase2 = normEarnings_phase2[opt_index_phase2][0]
optimal_bid_index_phase2 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase2,env_array)[1][0]
optimal_bid_phase2 = bids[int(optimal_bid_index_phase2)]

opt_index_phase3 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase3,env_array)[0][0])
opt_phase3 = normEarnings_phase3[opt_index_phase3][0]
optimal_bid_index_phase3 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase3,env_array)[1][0]
optimal_bid_phase3 = bids[int(optimal_bid_index_phase3)]


for e in tqdm(range(n_experiments)):
  env_swucb = deepcopy(env_array[0])
  env_cusum = deepcopy(env_array[0])
  swucb_learner = SWUCB_Learner(n_arms = n_prices, window_size = int(T/3))
  cusum_learner = CUSUM_UCB_Learner(n_arms = n_prices, M = M, eps = eps, h = h)
  for t in range(0, T):

    pulled_arm = swucb_learner.pull_arm()
    reward = env_swucb.round(pulled_arm)
    swucb_learner.update(pulled_arm, reward)

    pulled_arm = cusum_learner.pull_arm()
    reward = env_cusum.round(pulled_arm)
    cusum_learner.update(pulled_arm, reward)

  swucb_rewards_per_experiments.append(swucb_learner.collected_rewards)
  cusum_rewards_per_experiments.append(cusum_learner.collected_rewards)


swucb_rewards_per_experiments = np.array(swucb_rewards_per_experiments)
cusum_rewards_per_experiments = np.array(cusum_rewards_per_experiments)

fig, axs = plt.subplots(2,2,figsize=(14,7))

opt_phase1 = opt_phase1 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
opt_phase2 = opt_phase2 * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
opt_phase3 = opt_phase3 * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)
opt = np.ones([T]) 
opt[:int(T/3)] = opt[:int(T/3)] * opt_phase1 
opt[int(T/3):2*int(T/3)] = opt[int(T/3):2*int(T/3)]* opt_phase2 
opt[2*int(T/3):] = opt[2*int(T/3):] * opt_phase3

swucb_rewards_per_experiments[:int(T/3)] = swucb_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
swucb_rewards_per_experiments[int(T/3):2*int(T/3)] = swucb_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
swucb_rewards_per_experiments[2*int(T/3):] = swucb_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)

cusum_rewards_per_experiments[:int(T/3)] = cusum_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
cusum_rewards_per_experiments[int(T/3):2*int(T/3)] = cusum_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
cusum_rewards_per_experiments[2*int(T/3):] = cusum_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(swucb_rewards_per_experiments, axis = 0)), 'r')
axs[0][0].plot(np.cumsum(np.mean(cusum_rewards_per_experiments, axis = 0)), 'm')

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(swucb_rewards_per_experiments, axis = 0)), 'b')
axs[0][0].plot(np.cumsum(np.std(cusum_rewards_per_experiments, axis = 0)), 'c')

axs[0][0].plot(np.cumsum(np.mean(opt - swucb_rewards_per_experiments, axis = 0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(opt - cusum_rewards_per_experiments, axis = 0)), 'y')

axs[0][0].legend(["Reward SWUCB","Reward CUSUM","Std SWUCB","Std CUSUM","Regret SWUCB","Regret CUSUM"])
axs[0][0].set_title("Cumulative SWUCB vs CUSUM")


axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Regret")
axs[0][1].plot(np.mean(swucb_rewards_per_experiments, axis = 0), 'r')
axs[0][1].plot(np.mean(cusum_rewards_per_experiments, axis = 0), 'm')
axs[0][1].legend(["Reward SWUCB", "Reward CUSUM"])
axs[0][1].set_title("Instantaneous Reward SWUCB vs CUSUM")

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[1][0].plot(np.std(swucb_rewards_per_experiments, axis = 0), 'b')   
axs[1][0].plot(np.std(cusum_rewards_per_experiments, axis = 0), 'c')
axs[1][0].legend(["Std SWUCB","Std CUSUM"])
axs[1][0].set_title("Instantaneous Std SWUCB VS CUSUM")

axs[1][1].plot(np.mean(opt - swucb_rewards_per_experiments, axis = 0), 'g')
axs[1][1].plot(np.mean(opt - cusum_rewards_per_experiments, axis = 0), 'y')
axs[1][1].legend(["Regret SWUCB","Regret CUSUM"])
axs[1][1].set_title("Instantaneous Regret SWUCB vs CUSUM")


plt.show()

