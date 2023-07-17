from Classes.learners import Learner, EXP3_Learner, UCB1_Learner
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




#EXPERIMENT BEGIN
T = 365

n_experiments = 10

exp_rewards_per_experiments = []
ucb1_rewards_per_experiments = []

opt_index = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][0])
print(opt_index)
opt = normEarnings[opt_index][0]
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][0]
optimal_bid = bids[int(optimal_bid_index)]
print(opt)

for e in tqdm(range(n_experiments)):
  env = env_array[0]
  exp_learner = EXP3_Learner(n_arms = n_prices, gamma = 0.1)
  ucb1_learner = UCB1_Learner(n_arms = n_prices)
  for t in range(0, T):
    pulled_arm = exp_learner.pull_arm()
    reward = env.round(pulled_arm)
    exp_learner.update(pulled_arm, reward)
    
    pulled_arm = ucb1_learner.pull_arm()
    reward = env.round(pulled_arm) 
    ucb1_learner.update(pulled_arm, reward)


  exp_rewards_per_experiments.append(exp_learner.collected_rewards)
  ucb1_rewards_per_experiments.append(ucb1_learner.collected_rewards)

#num_arms_pulled = np.array(list(map(lambda x: len(x),ts_learner.reward_per_arm)))
#learned_optimal_price_index = np.argmax(num_arms_pulled)


exp_rewards_per_experiments = np.array(exp_rewards_per_experiments)
ucb1_rewards_per_experiments = np.array(ucb1_rewards_per_experiments)

fig, axs = plt.subplots(2,2,figsize=(14,7))

opt = opt * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)
exp_rewards_per_experiments = exp_rewards_per_experiments * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)
ucb1_rewards_per_experiments = ucb1_rewards_per_experiments * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(exp_rewards_per_experiments, axis = 0)), 'r')
axs[0][0].plot(np.cumsum(np.mean(ucb1_rewards_per_experiments, axis = 0)), 'm')


#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(exp_rewards_per_experiments, axis = 0)), 'b')   
axs[0][0].plot(np.cumsum(np.std(ucb1_rewards_per_experiments, axis = 0)), 'c')

axs[0][0].plot(np.cumsum(np.mean(opt - exp_rewards_per_experiments, axis = 0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiments, axis = 0)), 'y')

axs[0][0].legend(["Reward TS", "Reward UCB1","Std TS","Std UCB1","Regret TS","Regret UCB1"])
axs[0][0].set_title("Cumulative TS vs UCB1")



axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Regret")
axs[0][1].plot(np.mean(exp_rewards_per_experiments, axis = 0), 'r')
axs[0][1].plot(np.mean(ucb1_rewards_per_experiments, axis = 0), 'm')
axs[0][1].legend(["Reward TS", "Reward UCB1"])
axs[0][1].set_title("Instantaneous Reward TS vs UCB1")


#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[1][0].plot(np.std(exp_rewards_per_experiments, axis = 0), 'b')   
axs[1][0].plot(np.std(ucb1_rewards_per_experiments, axis = 0), 'c')
axs[1][0].legend(["Std TS","Std UCB1"])
axs[1][0].set_title("Instantaneous Std TS vs UCB1")


axs[1][1].plot(np.mean(opt - exp_rewards_per_experiments, axis = 0), 'g')
axs[1][1].plot(np.mean(opt - ucb1_rewards_per_experiments, axis = 0), 'y')
axs[1][1].legend(["Regret TS","Regret UCB1"])
axs[1][1].set_title("Instantaneous Regret TS vs UCB1")

plt.show()
