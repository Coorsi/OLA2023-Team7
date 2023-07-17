from Classes.learners import Learner,TS_Learner,UCB1_Learner,SWTS_Learner,SWUCB_Learner
from Classes.enviroment import Non_Stationary_Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
conversion_rate_phase2 =  np.array([[0.85, 0.87, 0.75],  # 1*price
                                    [0.76, 0.78, 0.40],  # 2*price
                                    [0.59, 0.62, 0.27],  # 3*price
                                    [0.46, 0.48, 0.19],  # 4*price
                                    [0.15, 0.17, 0.09]   # 5*price
                                    ])

                                      #C1   C2   C3
conversion_rate_phase3 =  np.array([[0.95, 0.97, 0.80],  # 1*price
                                    [0.88, 0.90, 0.48],  # 2*price
                                    [0.65, 0.68, 0.31],  # 3*price
                                    [0.51, 0.54, 0.23],  # 4*price
                                    [0.18, 0.20, 0.12]   # 5*price
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
  env_array.append(Non_Stationary_Environment(n_prices, np.array([normEarnings_phase1[:,c], normEarnings_phase2[:,c], normEarnings_phase3[:,c]]), c, T))

#EXPERIMENT BEGIN

n_experiments = 100

swucb_rewards_per_experiments = []

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
  env = env_array[0]
  swucb_learner = SWUCB_Learner(n_arms = n_prices, window_size = int(T/3))
  for t in range(0, T):

    pulled_arm = swucb_learner.pull_arm()
    reward = env.round(pulled_arm)
    swucb_learner.update(pulled_arm, reward)

  swucb_rewards_per_experiments.append(swucb_learner.collected_rewards)



swucb_rewards_per_experiments = np.array(swucb_rewards_per_experiments)

fig, axs = plt.subplots(2,2,figsize=(14,7))

opt_phase1 = opt_phase1 * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
opt_phase2 = opt_phase2 * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
opt_phase3 = opt_phase3 * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)
opt = np.ones([1,T]) 
opt[:int(T/3)] = opt[:int(T/3)] * opt_phase1 
opt[int(T/3):2*int(T/3)] = opt[int(T/3):2*int(T/3)]* opt_phase2 
opt[2*int(T/3):] = opt[2*int(T/3):] * opt_phase3

swucb_rewards_per_experiments[:int(T/3)] = swucb_rewards_per_experiments[:int(T/3)] * env_array[0].n(optimal_bid_phase1) - env_array[0].cc(optimal_bid_phase1)
swucb_rewards_per_experiments[int(T/3):2*int(T/3)] = swucb_rewards_per_experiments[int(T/3):2*int(T/3)] * env_array[0].n(optimal_bid_phase2) - env_array[0].cc(optimal_bid_phase2)
swucb_rewards_per_experiments[2*int(T/3):] = swucb_rewards_per_experiments[2*int(T/3):] * env_array[0].n(optimal_bid_phase3) - env_array[0].cc(optimal_bid_phase3)


axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(swucb_rewards_per_experiments, axis = 0)), 'r')


#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(swucb_rewards_per_experiments, axis = 0)), 'b')

axs[0][0].plot(np.cumsum(np.mean(opt - swucb_rewards_per_experiments, axis = 0)), 'g')

axs[0][0].legend(["Reward SWUCB","Std SWUCB","Regret SWUCB"])
axs[0][0].set_title("Cumulative SWUCB")


axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Regret")
axs[0][1].plot(np.mean(swucb_rewards_per_experiments, axis = 0), 'r')

axs[0][1].plot(np.std(swucb_rewards_per_experiments, axis = 0), 'b')

axs[0][1].plot(np.mean(opt - swucb_rewards_per_experiments, axis = 0), 'g')

axs[0][1].legend(["Reward SWUCB","Std SWUCB","Regret SWUCB"])
axs[0][1].set_title("Instantaneous SWUCB")


axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Regret")
axs[1][0].plot(np.std(swucb_rewards_per_experiments, axis = 0), 'b')

axs[1][0].legend(["Std SWUCB"])
axs[1][0].set_title("Instantaneous Std SWUCB")


axs[1][1].plot(np.mean(opt - swucb_rewards_per_experiments, axis = 0), 'g')
axs[1][1].legend(["Regret SWUCB"])
axs[1][1].set_title("Instantaneous Regret SWUCB")

plt.show()

