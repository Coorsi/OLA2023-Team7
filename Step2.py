from Classes.learners import Learner,GPTS_Learner,GPUCB_Learner
from Classes.enviroment import Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from tqdm.autonotebook import tqdm



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

earnings = earnings - np.min(earnings)

earnings = earnings / np.max(earnings)

env_array = []
for c in classes:
  env_array.append(Environment(n_prices, earnings[:,c], c))





#EXPERIMENT BEGIN
T = 365

n_experiments = 100

gpts_num_click_per_experiments = []
gpts_cum_cost_per_experiments = []

gpucb_num_click_per_experiments = []
gpucb_cum_cost_per_experiments = []

opt_index = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][0])
print(opt_index)
opt = earnings[opt_index][0]
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][0]
optimal_bid = bids[int(optimal_bid_index)]
print(opt)

for e in range(n_experiments):
  print(e)
  env = env_array[0]
  gpts_learner_n = GPTS_Learner(n_arms = n_bids, arms = bids)
  gpts_learner_cc = GPTS_Learner(n_arms = n_bids, arms = bids)
  gpucb_learner_n = GPUCB_Learner(n_arms = n_bids, arms = bids)
  gpucb_learner_cc = GPUCB_Learner(n_arms = n_bids, arms = bids)

  for t in tqdm(range(T)):
    pulled_arm = gpts_learner_n.pull_arm()
    reward = env.draw_n(bids[pulled_arm],1) # 1 is std
    gpts_learner_n.update(pulled_arm, reward)
    pulled_arm = gpts_learner_cc.pull_arm()
    reward = env.draw_cc(bids[pulled_arm],1) # 1 is std
    gpts_learner_cc.update(pulled_arm, reward)
    
    pulled_arm = gpucb_learner_n.pull_arm()
    reward = env.draw_n(bids[pulled_arm],1) # 1 is std
    gpucb_learner_n.update(pulled_arm, reward)
    pulled_arm = gpucb_learner_cc.pull_arm()
    reward = env.draw_cc(bids[pulled_arm],1) # 1 is std
    gpucb_learner_cc.update(pulled_arm, reward)

  gpts_num_click_per_experiments.append(gpts_learner_n.collected_rewards)
  gpts_cum_cost_per_experiments.append(gpts_learner_cc.collected_rewards)
  gpucb_num_click_per_experiments.append(gpucb_learner_n.collected_rewards)
  gpucb_cum_cost_per_experiments.append(gpucb_learner_cc.collected_rewards)

gpts_num_click_per_experiments = np.array(gpts_num_click_per_experiments)
gpts_cum_cost_per_experiments = np.array(gpts_cum_cost_per_experiments)
gpucb_num_click_per_experiments = np.array(gpucb_num_click_per_experiments)
gpucb_cum_cost_per_experiments = np.array(gpucb_cum_cost_per_experiments)


#print("gpts_num_click_per_experiments",gpts_num_click_per_experiments)
#print("gpts_cum_cost_per_experiments",gpts_cum_cost_per_experiments)

#print("gpucb_num_click_per_experiments",gpucb_num_click_per_experiments)
#print("gpucb_cum_cost_per_experiments",gpucb_cum_cost_per_experiments)

fig, axs = plt.subplots(1,2,figsize=(14,7))

opt_reward = opt * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)
gpts_rewards_per_experiments = opt * gpts_num_click_per_experiments - gpts_cum_cost_per_experiments
gpucb_rewards_per_experiments = opt * gpucb_num_click_per_experiments - gpucb_cum_cost_per_experiments

axs[0].set_xlabel("t")
axs[0].set_ylabel("Regret")
axs[0].plot(np.cumsum(np.mean(gpts_rewards_per_experiments, axis = 0)), 'r')
axs[0].plot(np.cumsum(np.mean(gpucb_rewards_per_experiments, axis = 0)), 'm')


#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0].plot(np.cumsum(np.std(gpts_rewards_per_experiments, axis = 0)), 'b')   
axs[0].plot(np.cumsum(np.std(gpucb_rewards_per_experiments, axis = 0)), 'c')

axs[0].plot(np.cumsum(np.mean(opt_reward - gpts_rewards_per_experiments, axis = 0)), 'g')
axs[0].plot(np.cumsum(np.mean(opt_reward - gpucb_rewards_per_experiments, axis = 0)), 'y')

axs[0].legend(["Reward GPTS", "Reward GPUCB","Std GPTS","Std GPUCB","Regret GPTS","Regret GPUCB"])
axs[0].set_title("Cumulative GPTS vs GPUCB")



axs[1].set_xlabel("t")
axs[1].set_ylabel("Regret")
axs[1].plot(np.mean(gpts_rewards_per_experiments, axis = 0), 'r')
axs[1].plot(np.mean(gpucb_rewards_per_experiments, axis = 0), 'm')


#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[1].plot(np.std(gpts_rewards_per_experiments, axis = 0), 'b')   
axs[1].plot(np.std(gpucb_rewards_per_experiments, axis = 0), 'c')

axs[1].plot(np.mean(opt_reward - gpts_rewards_per_experiments, axis = 0), 'g')
axs[1].plot(np.mean(opt_reward - gpucb_rewards_per_experiments, axis = 0), 'y')

axs[1].legend(["Reward GPTS", "Reward GPUCB","Std GPTS","Std GPUCB","Regret GPTS","Regret GPUCB"])
axs[1].set_title("Instantaneous GPTS vs GPUCB")

plt.show()
