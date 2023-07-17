from Classes.learners import Learner,TS_Learner,UCB1_Learner,GPTS_Learner,GPUCB_Learner
from Classes.enviroment import Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


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

#EXPERIMENT BEGIN FOR ESTIMATING THE OPTIMAL PRICE 
T = 365

n_experiments = 100

ts_rewards_per_experiments_c1 = []
ts_rewards_per_experiments_c2 = []
ts_rewards_per_experiments_c3 = []
ts_rewards_per_experiments = []

#with multiple classes each class is optimized independently, the total reward is the sum of all rewards

for e in tqdm(range(n_experiments)):
  env_c1 = env_array[0]
  env_c2 = env_array[1]
  env_c3 = env_array[2]
  ts_learner_c1 = TS_Learner(n_arms = n_prices)
  ts_learner_c2 = TS_Learner(n_arms = n_prices)
  ts_learner_c3 = TS_Learner(n_arms = n_prices)
  for t in range(0, T):
    pulled_arm_c1 = ts_learner_c1.pull_arm()
    pulled_arm_c2 = ts_learner_c2.pull_arm()
    pulled_arm_c3 = ts_learner_c3.pull_arm()
    reward_c1 = env_c1.round(pulled_arm_c1) 
    reward_c2 = env_c2.round(pulled_arm_c2)
    reward_c3 = env_c3.round(pulled_arm_c3)
    ts_learner_c1.update(pulled_arm_c1, reward_c1)
    ts_learner_c2.update(pulled_arm_c2, reward_c2)
    ts_learner_c3.update(pulled_arm_c3, reward_c3)

  ts_rewards_per_experiments_c1.append(ts_learner_c1.collected_rewards)
  ts_rewards_per_experiments_c2.append(ts_learner_c2.collected_rewards)
  ts_rewards_per_experiments_c3.append(ts_learner_c3.collected_rewards)

"""
num_arms_pulled = np.array(list(map(lambda x: len(x),ts_learner_c1.reward_per_arm)))
learned_optimal_price_index = np.argmax(num_arms_pulled)
print(len(ts_learner_c1.reward_per_arm[learned_optimal_price_index])) """

ts_rewards_per_experiments = np.array(ts_rewards_per_experiments)
ts_rewards_per_experiments_c1 = np.array(ts_rewards_per_experiments_c1)
ts_rewards_per_experiments_c2 = np.array(ts_rewards_per_experiments_c2)
ts_rewards_per_experiments_c3 = np.array(ts_rewards_per_experiments_c3)

#best price for c1
num_arms_pulled_c1 = np.array(list(map(lambda x: len(x),ts_learner_c1.reward_per_arm)))
learned_optimal_price_index_c1 = np.argmax(num_arms_pulled_c1)

#best price for c2
num_arms_pulled_c2 = np.array(list(map(lambda x: len(x),ts_learner_c2.reward_per_arm)))
learned_optimal_price_index_c2 = np.argmax(num_arms_pulled_c2)

#best price for c3
num_arms_pulled_c3 = np.array(list(map(lambda x: len(x),ts_learner_c3.reward_per_arm)))
learned_optimal_price_index_c3 = np.argmax(num_arms_pulled_c3)

#EXPERIMENT BEGIN FOR ESTIMATING BEST BID
T = 365

n_experiments = 100
noise_std = 1

gpts_reward = []
gpts_reward_c1 = []
gpts_reward_c2 = []
gpts_reward_c3 = []

gpucb_reward = []
gpucb_reward_c1 = []
gpucb_reward_c2 = []
gpucb_reward_c3 = []

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

for e in range(n_experiments):
  print(e)
  gpts_learner_c1 = GPTS_Learner(n_arms = n_bids, arms = bids)
  gpts_learner_c2 = GPTS_Learner(n_arms = n_bids, arms = bids)
  gpts_learner_c3 = GPTS_Learner(n_arms = n_bids, arms = bids)

  gpucb_learner_c1 = GPUCB_Learner(n_arms = n_bids, arms = bids)
  gpucb_learner_c2 = GPUCB_Learner(n_arms = n_bids, arms = bids)
  gpucb_learner_c3 = GPUCB_Learner(n_arms = n_bids, arms = bids)

  for t in tqdm(range(T)):
    #gpts
    pulled_arm_c1 = gpts_learner_c1.pull_arm()
    reward_c1 = env_c1.draw_n(bids[pulled_arm_c1],noise_std) * earnings[learned_optimal_price_index_c1][0] - env_c1.draw_cc(bids[pulled_arm_c1],noise_std) # 1 is std
    gpts_learner_c1.update(pulled_arm_c1, reward_c1)

    pulled_arm_c2 = gpts_learner_c2.pull_arm()
    reward_c2 = env_c2.draw_n(bids[pulled_arm_c2],noise_std) * earnings[learned_optimal_price_index_c2][1] - env_c2.draw_cc(bids[pulled_arm_c2],noise_std) # 1 is std
    gpts_learner_c2.update(pulled_arm_c2, reward_c2)

    pulled_arm_c3 = gpts_learner_c3.pull_arm()
    reward_c3 = env_c3.draw_n(bids[pulled_arm_c3],noise_std) * earnings[learned_optimal_price_index_c3][2] - env_c3.draw_cc(bids[pulled_arm_c3],noise_std) # 1 is std
    gpts_learner_c3.update(pulled_arm_c3, reward_c3)


    #gpucb
    pulled_arm_c1 = gpucb_learner_c1.pull_arm()
    reward_c1 = env_c1.draw_n(bids[pulled_arm_c1],noise_std) * earnings[learned_optimal_price_index_c1][0] - env_c1.draw_cc(bids[pulled_arm_c1],noise_std)# 1 is std
    gpucb_learner_c1.update(pulled_arm_c1, reward_c1)

    pulled_arm_c2 = gpucb_learner_c2.pull_arm()
    reward_c2 = env_c2.draw_n(bids[pulled_arm_c2],noise_std) * earnings[learned_optimal_price_index_c2][1] - env_c2.draw_cc(bids[pulled_arm_c2],noise_std)# 1 is std
    gpucb_learner_c2.update(pulled_arm_c2, reward_c2)

    pulled_arm_c3 = gpucb_learner_c3.pull_arm()
    reward_c3 = env_c3.draw_n(bids[pulled_arm_c3],noise_std) * earnings[learned_optimal_price_index_c3][2] - env_c3.draw_cc(bids[pulled_arm_c3],noise_std)# 1 is std
    gpucb_learner_c3.update(pulled_arm_c3, reward_c3)

  gpts_reward_c1.append(gpts_learner_c1.collected_rewards)
  gpts_reward_c2.append(gpts_learner_c2.collected_rewards)
  gpts_reward_c3.append(gpts_learner_c3.collected_rewards)
  gpucb_reward_c1.append(gpucb_learner_c1.collected_rewards)
  gpucb_reward_c2.append(gpucb_learner_c2.collected_rewards)
  gpucb_reward_c3.append(gpucb_learner_c3.collected_rewards)

gpts_reward_c1 = np.array(gpts_reward_c1)
gpts_reward_c2 = np.array(gpts_reward_c2)
gpts_reward_c3 = np.array(gpts_reward_c3)
gpucb_reward_c1 = np.array(gpucb_reward_c1)
gpucb_reward_c2 = np.array(gpucb_reward_c2)
gpucb_reward_c3 = np.array(gpucb_reward_c3)

gpts_reward = np.array(gpts_reward)
gpts_reward = gpts_reward_c1 + gpts_reward_c2 + gpts_reward_c3

gpucb_reward = np.array(gpucb_reward)
gpucb_reward = gpucb_reward_c1 + gpucb_reward_c2 + gpucb_reward_c3

opt_reward_1 = opt1 * env_array[0].n(optimal_bid_1) - env_array[0].cc(optimal_bid_1) 
opt_reward_2 = opt2 * env_array[1].n(optimal_bid_2) - env_array[1].cc(optimal_bid_2)
opt_reward_3 = opt3 * env_array[2].n(optimal_bid_3) - env_array[2].cc(optimal_bid_3)

#total regret is the sum of all regrets
tot_regret_gpts = []
tot_regret_gpts = np.array(tot_regret_gpts)
tot_regret_gpucb = []
tot_regret_gpucb = np.array(tot_regret_gpucb)

tot_regret_gpts = (opt_reward_1 - gpts_reward_c1) + (opt_reward_2 - gpts_reward_c2) + (opt_reward_3 - gpts_reward_c3)
tot_regret_gpucb = (opt_reward_1 - gpucb_reward_c1) + (opt_reward_2 - gpucb_reward_c2) + (opt_reward_3 - gpucb_reward_c3)

#plot
fig, axs = plt.subplots(2,2,figsize=(24,12))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(gpts_reward, axis = 0)), 'r')
axs[0][0].plot(np.cumsum(np.mean(gpucb_reward, axis = 0)), 'm')

#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(gpts_reward, axis = 0)), 'b')   
axs[0][0].plot(np.cumsum(np.std(gpucb_reward, axis = 0)), 'c')

axs[0][0].plot(np.cumsum(np.mean(tot_regret_gpts, axis = 0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(tot_regret_gpucb, axis = 0)), 'y')

axs[0][0].legend(["Reward GPTS", "Reward GPUCB","Std GPTS","Std GPUCB","Regret GPTS","Regret GPUCB"])
axs[0][0].set_title("Cumulative GPTS vs GPUCB")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.mean(gpts_reward, axis = 0), 'r')
axs[0][1].plot(np.mean(gpucb_reward, axis = 0), 'm')
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
axs[1][1].plot(np.std(gpts_reward, axis = 0), 'b')   
axs[1][1].plot(np.std(gpucb_reward, axis = 0), 'c')
axs[1][1].legend(["Std GPTS","Std GPUCB"])
axs[1][1].set_title("Instantaneous Reward GPTS vs GPUCB")

plt.show()
#print(gpts_reward)
#print(gpucb_reward)
#print(opt_reward)