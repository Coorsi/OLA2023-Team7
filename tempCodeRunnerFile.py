from Classes.learners import Learner,TS_Learner,UCB1_Learner
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




#EXPERIMENT BEGIN FOR ESTIMATING THE OPTIMAL PRICE 
T = 365

n_experiments = 1000

ts_rewards_per_experiments = []

for e in tqdm(range(n_experiments)):
  env = env_array[0]
  ts_learner = TS_Learner(n_arms = n_prices)
  for t in range(0, T):
    pulled_arm = ts_learner.pull_arm()
    reward = env.round(pulled_arm)
    ts_learner.update(pulled_arm, reward)

  ts_rewards_per_experiments.append(ts_learner.collected_rewards)

num_arms_pulled = np.array(list(map(lambda x: len(x),ts_learner.reward_per_arm)))
learned_optimal_price_index = np.argmax(num_arms_pulled)
print(len(ts_learner.reward_per_arm[learned_optimal_price_index]))