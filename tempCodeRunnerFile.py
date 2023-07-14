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
print(earnings)
normEarnings = earnings.copy()
print(normEarnings)
normEarnings = normEarnings - np.min(normEarnings)

normEarnings = normEarnings / np.max(normEarnings)

env_array = []
for c in classes:
  env_array.append(Environment(n_prices, normEarnings[:,c], c))

print(earnings)
print(normEarnings)

