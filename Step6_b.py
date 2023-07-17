from Classes.learners import Learner,TS_Learner,UCB1_Learner,SWTS_Learner,SWUCB_Learner,CUSUM_UCB_Learner, EXP3_Learner
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

                                      #C1   C2   C3
conversion_rate_phase4 =  np.array([[0.91, 0.88, 0.79],  # 1*price
                                    [0.83, 0.71, 0.44],  # 2*price
                                    [0.66, 0.68, 0.30],  # 3*price
                                    [0.52, 0.43, 0.27],  # 4*price
                                    [0.21, 0.17, 0.11]   # 5*price
                                    ])

                                      #C1   C2   C3
conversion_rate_phase5 =  np.array([[0.92, 0.96, 0.72],  # 1*price
                                    [0.80, 0.90, 0.44],  # 2*price
                                    [0.75, 0.74, 0.28],  # 3*price
                                    [0.52, 0.57, 0.23],  # 4*price
                                    [0.26, 0.18, 0.07]   # 5*price
                                    ])

earnings_phase1 = np.zeros([5,3]) # conv_rate * margin
earnings_phase2 = np.zeros([5,3]) # conv_rate * margin
earnings_phase3 = np.zeros([5,3]) # conv_rate * margin
earnings_phase4 = np.zeros([5,3]) # conv_rate * margin
earnings_phase5 = np.zeros([5,3]) # conv_rate * margin

for row in range(5):
  earnings_phase1[row,:] = conversion_rate_phase1[row,:] * margins[row]
  earnings_phase2[row,:] = conversion_rate_phase2[row,:] * margins[row]
  earnings_phase3[row,:] = conversion_rate_phase3[row,:] * margins[row]
  earnings_phase4[row,:] = conversion_rate_phase3[row,:] * margins[row]
  earnings_phase5[row,:] = conversion_rate_phase3[row,:] * margins[row]

normEarnings_phase1 = earnings_phase1.copy()
normEarnings_phase1 = normEarnings_phase1 - np.min(normEarnings_phase1)
normEarnings_phase1 = normEarnings_phase1 / np.max(normEarnings_phase1)

normEarnings_phase2 = earnings_phase2.copy()
normEarnings_phase2 = normEarnings_phase2 - np.min(normEarnings_phase2)
normEarnings_phase2 = normEarnings_phase2 / np.max(normEarnings_phase2)

normEarnings_phase3 = earnings_phase3.copy()
normEarnings_phase3 = normEarnings_phase3 - np.min(normEarnings_phase3)
normEarnings_phase3 = normEarnings_phase3 / np.max(normEarnings_phase3)

normEarnings_phase4 = earnings_phase4.copy()
normEarnings_phase4 = normEarnings_phase4 - np.min(normEarnings_phase4)
normEarnings_phase4 = normEarnings_phase4 / np.max(normEarnings_phase4)

normEarnings_phase5 = earnings_phase5.copy()
normEarnings_phase5 = normEarnings_phase5 - np.min(normEarnings_phase5)
normEarnings_phase5 = normEarnings_phase5 / np.max(normEarnings_phase5)

env_array = []
T = 365
for c in classes:
  env_array.append(Non_Stationary_Environment(n_prices, np.array([normEarnings_phase1[:,c], normEarnings_phase2[:,c], normEarnings_phase3[:,c], normEarnings_phase4[:,c], normEarnings_phase5[:,c]]), c, T, 1))

#EXPERIMENT BEGIN

n_experiments = 100

M = 100 #number of steps to obtain reference point in change detection (for CUSUM)
eps = 0.1 #epsilon for deviation from reference point in change detection (for CUSUM)
h = np.log(T)*2 #threshold for change detection (for CUSUM)

swucb_rewards_per_experiments = []
cusum_rewards_per_experiments = []
exp3_rewards_per_experiments = []

opt_index_phase1 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase1,env_array)[0][0])
opt_phase1 = normEarnings_phase1[opt_index_phase1][0]
optimal_bid_index_phase1 = clairvoyant(classes,bids,prices, margins,conversion_rate_phase1,env_array)[1][0]
optimal_bid_phase1 = bids[int(optimal_bid_index_phase1)] #we consider the same bid (?)

opt_index_phase2 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase2,env_array)[0][0])
opt_phase2 = normEarnings_phase2[opt_index_phase2][0]

opt_index_phase3 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase3,env_array)[0][0])
opt_phase3 = normEarnings_phase3[opt_index_phase3][0]

opt_index_phase4 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase4,env_array)[0][0])
opt_phase4 = normEarnings_phase3[opt_index_phase4][0]

opt_index_phase5 = int(clairvoyant(classes,bids,prices, margins,conversion_rate_phase5,env_array)[0][0])
opt_phase5 = normEarnings_phase3[opt_index_phase5][0]

for e in tqdm(range(n_experiments)):
  env_swucb = deepcopy(env_array[0])
  env_cusum = deepcopy(env_array[0])
  env_exp3 = deepcopy(env_array[0])
  swucb_learner = SWUCB_Learner(n_arms = n_prices, window_size = int(T/3))
  cusum_learner = CUSUM_UCB_Learner(n_arms = n_prices, M = M, eps = eps, h = h)
  exp3_learner = EXP3_Learner(n_arms = n_prices, gamma = 0.01)
  for t in range(0, T):

    pulled_arm = swucb_learner.pull_arm()
    reward = env_swucb.round(pulled_arm)
    swucb_learner.update(pulled_arm, reward)

    pulled_arm = cusum_learner.pull_arm()
    reward = env_cusum.round(pulled_arm)
    cusum_learner.update(pulled_arm, reward)

    pulled_arm = exp3_learner.pull_arm()
    reward = env_exp3.round(pulled_arm)
    exp3_learner.update(pulled_arm, reward)

  swucb_rewards_per_experiments.append(swucb_learner.collected_rewards)
  cusum_rewards_per_experiments.append(cusum_learner.collected_rewards)
  exp3_rewards_per_experiments.append(exp3_learner.collected_rewards)

swucb_rewards_per_experiments = np.array(swucb_rewards_per_experiments)
cusum_rewards_per_experiments = np.array(cusum_rewards_per_experiments)
exp3_rewards_per_experiments = np.array(exp3_rewards_per_experiments)

fig, axs = plt.subplots(2,2,figsize=(14,7))

opt_phase1 = opt_phase1 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
opt_phase2 = opt_phase2 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
opt_phase3 = opt_phase3 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
opt_phase4 = opt_phase4 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
opt_phase5 = opt_phase5 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)

opt = np.ones([T])
size_phases = int(T / 20)

opt[:int(T/20)] = opt[:int(T/20)] * opt_phase1 
for i in range (1, size_phases - 1):
  if (i%5 == 0):
    opt[(i-1)*int(T/20):i*int(T/20)] = opt[(i-1)*int(T/20):i*int(T/20)]* opt_phase1
  elif (i%5 == 1):
    opt[(i-1)*int(T/20):i*int(T/20)] = opt[(i-1)*int(T/20):i*int(T/20)]* opt_phase2
  elif(i%5 == 2):
    opt[(i-1)*int(T/20):i*int(T/20)] = opt[(i-1)*int(T/20):i*int(T/20)]* opt_phase3
  elif (i%5 == 3):
    opt[(i-1)*int(T/20):i*int(T/20)] = opt[(i-1)*int(T/20):i*int(T/20)]* opt_phase4
  elif(i%5 == 4):
    opt[(i-1)*int(T/20):i*int(T/20)] = opt[(i-1)*int(T/20):i*int(T/20)]* opt_phase5 

opt[19*int(T/20):] = opt[19*int(T/20):] * opt_phase5

swucb_rewards_per_experiments[:int(T/20)] = swucb_rewards_per_experiments[:int(T/20)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1) 
for i in range (1, size_phases - 1):
  if (i%5 == 0):
    swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif (i%5 == 1):
    swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif(i%5 == 2):
    swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif (i%5 == 3):
    swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif(i%5 == 4):
    swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = swucb_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)

swucb_rewards_per_experiments[19*int(T/20):] = swucb_rewards_per_experiments[19*int(T/20):] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)


cusum_rewards_per_experiments[:int(T/20)] = cusum_rewards_per_experiments[:int(T/20)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1) 
for i in range (1, size_phases - 1):
  if (i%5 == 0):
    cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif (i%5 == 1):
    cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif(i%5 == 2):
    cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif (i%5 == 3):
    cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif(i%5 == 4):
    cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = cusum_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)

cusum_rewards_per_experiments[19*int(T/20):] = cusum_rewards_per_experiments[19*int(T/20):] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)

exp3_rewards_per_experiments[:int(T/20)] = exp3_rewards_per_experiments[:int(T/20)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1) 
for i in range (1, size_phases - 1):
  if (i%5 == 0):
    exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif (i%5 == 1):
    exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif(i%5 == 2):
    exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif (i%5 == 3):
    exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
  elif(i%5 == 4):
    exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)] = exp3_rewards_per_experiments[(i-1)*int(T/20):i*int(T/20)]* env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)

exp3_rewards_per_experiments[19*int(T/20):] = exp3_rewards_per_experiments[19*int(T/20):] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)


axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(swucb_rewards_per_experiments, axis = 0)), 'darkorange')
axs[0][0].plot(np.cumsum(np.mean(cusum_rewards_per_experiments, axis = 0)), 'darkgreen')
axs[0][0].plot(np.cumsum(np.mean(exp3_rewards_per_experiments, axis = 0)), 'darkblue')

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(swucb_rewards_per_experiments, axis = 0)), 'gold')
axs[0][0].plot(np.cumsum(np.std(cusum_rewards_per_experiments, axis = 0)), 'mediumturquoise')
axs[0][0].plot(np.cumsum(np.std(exp3_rewards_per_experiments, axis = 0)), 'mediumpurple')

axs[0][0].plot(np.cumsum(np.mean(opt - swucb_rewards_per_experiments, axis = 0)), 'lawngreen')
axs[0][0].plot(np.cumsum(np.mean(opt - cusum_rewards_per_experiments, axis = 0)), 'steelblue')
axs[0][0].plot(np.cumsum(np.mean(opt - exp3_rewards_per_experiments, axis = 0)), 'hotpink')

axs[0][0].legend(["Reward SWUCB","Reward CUSUM", "Reward EXP3", "Std SWUCB","Std CUSUM", "Std EXP3", "Regret SWUCB","Regret CUSUM", "Regret EXP3"])
axs[0][0].set_title("Cumulative SWUCB vs CUSUM vs EXP3")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Regret")
axs[0][1].plot(np.mean(swucb_rewards_per_experiments, axis = 0), 'r')
axs[0][1].plot(np.mean(cusum_rewards_per_experiments, axis = 0), 'm')
axs[0][1].plot(np.mean(exp3_rewards_per_experiments, axis = 0), 'b')
axs[0][1].legend(["Reward SWUCB", "Reward CUSUM", "Reward EXP3"])
axs[0][1].set_title("Instantaneous Reward SWUCB vs CUSUM vs EXP3")

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[1][0].plot(np.std(swucb_rewards_per_experiments, axis = 0), 'b')   
axs[1][0].plot(np.std(cusum_rewards_per_experiments, axis = 0), 'c')
axs[1][0].plot(np.std(exp3_rewards_per_experiments, axis = 0), 'r')
axs[1][0].legend(["Std SWUCB","Std CUSUM", "Std EXP3"])
axs[1][0].set_title("Instantaneous Std SWUCB VS CUSUM vs EXP3")

axs[1][1].plot(np.mean(opt - swucb_rewards_per_experiments, axis = 0), 'g')
axs[1][1].plot(np.mean(opt - cusum_rewards_per_experiments, axis = 0), 'y')
axs[1][1].plot(np.mean(opt - exp3_rewards_per_experiments, axis = 0), 'b')
axs[1][1].legend(["Regret SWUCB","Regret CUSUM", "Regret EXP3"])
axs[1][1].set_title("Instantaneous Regret SWUCB vs CUSUM vs EXP3")


plt.show()