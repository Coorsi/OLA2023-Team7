from Classes.learners import TS_Learner, GPTS_Learner, GPUCB_Learner
from Classes.enviroment import Environment
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
conversion_rate = np.array([[0.93,0.95,0.77], #1*price
                            [0.82,0.84,0.42], #2*price
                            [0.51,0.64,0.29], #3*price
                            [0.38,0.50,0.21], #4*price
                            [0.09,0.18,0.11]  #5*price
                            ]) 

earnings = np.zeros([5,3]) # conv_rate * margin
for row in range(5):
  earnings[row,:] = conversion_rate[row,:] * margins[row]

normEarnings = earnings.copy()
normEarnings = normEarnings - np.min(normEarnings)
normEarnings = normEarnings / np.max(normEarnings)

env_array = []
for c in classes:
  env_array.append(Environment(n_prices, normEarnings[:,c], c))

opt_index_1 = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][0])
opt_index_2 = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][1])
opt_index_3 = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][2])
#opt = normEarnings[opt_index][0]
#3 classes
opt1 = normEarnings[opt_index_1][0]
opt2 = normEarnings[opt_index_2][1]
opt3 = normEarnings[opt_index_3][2]

optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][0]
optimal_bid_1 = bids[int(optimal_bid_index)]
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][1]
optimal_bid_2 = bids[int(optimal_bid_index)]
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][2]
optimal_bid_3 = bids[int(optimal_bid_index)]
print(optimal_bid_1)
print(optimal_bid_2)
print(optimal_bid_3)
print('\n\n')

#EXPERIMENT BEGIN FOR ESTIMATING THE OPTIMAL PRICE
T = 100

n_experiments = 15
noise_std = 0.4

ts_rewards_per_experiments = [[], [], []]
gpts_rewards = [[], [], []]
gpucb_rewards = [[], [], []]

for e in tqdm(range(n_experiments)):
  ts_learners = [TS_Learner(n_arms=n_prices), TS_Learner(n_arms=n_prices), TS_Learner(n_arms=n_prices)]
  gpts_learners = [GPTS_Learner(n_arms = n_bids, arms = bids), GPTS_Learner(n_arms = n_bids, arms = bids),
                   GPTS_Learner(n_arms = n_bids, arms = bids)]
  gpucb_learners = [GPUCB_Learner(n_arms = n_bids, arms = bids), GPUCB_Learner(n_arms = n_bids, arms = bids),
                    GPUCB_Learner(n_arms = n_bids, arms = bids)]

  for t in range(0, T):
    for i in range(len(classes)):
      pulled_arm_price = ts_learners[i].pull_arm()
      reward = env_array[i].round(pulled_arm_price)
      sampled_normEarning = np.random.beta(ts_learners[i].beta_parameters[pulled_arm_price, 0],
                                           ts_learners[i].beta_parameters[pulled_arm_price, 1])
      # print(sampled_normEarning)
      ts_learners[i].update(pulled_arm_price, reward)

      pulled_arm_bid = gpts_learners[i].pull_arm()
      reward_tot = env_array[i].draw_n(bids[pulled_arm_bid],noise_std) * sampled_normEarning - env_array[i].draw_cc(
                                           bids[pulled_arm_bid],noise_std)
      gpts_learners[i].update(pulled_arm_bid, reward_tot)

      pulled_arm_bid = gpucb_learners[i].pull_arm()
      reward_tot = env_array[i].draw_n(bids[pulled_arm_bid],noise_std) * sampled_normEarning - env_array[i].draw_cc(
                                           bids[pulled_arm_bid],noise_std)
      gpucb_learners[i].update(pulled_arm_bid, reward_tot)

  for i in range(len(classes)):
    ts_rewards_per_experiments[i].append(ts_learners[i].collected_rewards)
    gpts_rewards[i].append(gpts_learners[i].collected_rewards)
    gpucb_rewards[i].append(gpucb_learners[i].collected_rewards)

for i in range(len(classes)):
  gpts_rewards[i] = np.array(gpts_rewards[i])
  gpucb_rewards[i] = np.array(gpucb_rewards[i])

gpts_reward_tot = gpts_rewards[0] + gpts_rewards[1] + gpts_rewards[2]
gpucb_reward_tot = gpucb_rewards[0] + gpucb_rewards[1] + gpucb_rewards[2]

opt_reward_1 = opt1 * env_array[0].n(optimal_bid_1) - env_array[0].cc(optimal_bid_1) 
opt_reward_2 = opt2 * env_array[1].n(optimal_bid_2) - env_array[1].cc(optimal_bid_2)
opt_reward_3 = opt3 * env_array[2].n(optimal_bid_3) - env_array[2].cc(optimal_bid_3)

#total regret is the sum of all regrets
tot_regret_gpts = (opt_reward_1 - gpts_rewards[0]) + (opt_reward_2 - gpts_rewards[1]) + (opt_reward_3 - gpts_rewards[2])
tot_regret_gpucb = (opt_reward_1 - gpucb_rewards[0]) + (opt_reward_2 - gpucb_rewards[1]) + (opt_reward_3 - gpucb_rewards[2])

#plot
fig, axs = plt.subplots(2,2,figsize=(24,12))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(gpts_reward_tot, axis = 0)), 'r')
axs[0][0].plot(np.cumsum(np.mean(gpucb_reward_tot, axis = 0)), 'm')

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(gpts_reward_tot, axis = 0)), 'b')
axs[0][0].plot(np.cumsum(np.std(gpucb_reward_tot, axis = 0)), 'c')

axs[0][0].plot(np.cumsum(np.mean(tot_regret_gpts, axis = 0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(tot_regret_gpucb, axis = 0)), 'y')

axs[0][0].legend(["Reward GPTS", "Reward GPUCB","Std GPTS","Std GPUCB","Regret GPTS","Regret GPUCB"])
axs[0][0].set_title("Cumulative GPTS vs GPUCB")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.mean(gpts_reward_tot, axis = 0), 'r')
axs[0][1].plot(np.mean(gpucb_reward_tot, axis = 0), 'm')
axs[0][1].legend(["Reward GPTS", "Reward GPUCB"])
axs[0][1].set_title("Instantaneous Reward GPTS vs GPUCB")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Regret")
axs[1][0].plot(np.mean(tot_regret_gpts, axis = 0), 'g')
axs[1][0].plot(np.mean(tot_regret_gpucb, axis = 0), 'y')
axs[1][0].legend(["Regret GPTS","Regret GPUCB"])
axs[1][0].set_title("Instantaneous Std GPTS vs GPUCB")

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Reward")
axs[1][1].plot(np.std(gpts_reward_tot, axis = 0), 'b')
axs[1][1].plot(np.std(gpucb_reward_tot, axis = 0), 'c')
axs[1][1].legend(["Std GPTS","Std GPUCB"])
axs[1][1].set_title("Instantaneous Reward GPTS vs GPUCB")

plt.show()
#print(gpts_reward)
#print(gpucb_reward)
#print(opt_reward)