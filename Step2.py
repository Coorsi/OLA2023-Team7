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
print(earnings)
#earnings = earnings - np.min(earnings)

#earnings = earnings / np.max(earnings)

env_array = []
for c in classes:
  env_array.append(Environment(n_prices, earnings[:,c], c))





#EXPERIMENT BEGIN
T = 365

n_experiments = 100
noise_std = 1

gpts_reward = []
gpucb_reward = []

print(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array))
opt_index = int(clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[0][0])
print(opt_index)
opt = earnings[opt_index][0] #0 bc only first Class
optimal_bid_index = clairvoyant(classes,bids,prices, margins,conversion_rate,env_array)[1][0]
optimal_bid = bids[int(optimal_bid_index)]
print(optimal_bid)
print(opt)

for e in range(n_experiments):
  print(e)
  env = env_array[0]
  gpts_learner = GPTS_Learner(n_arms = n_bids, arms = bids)

  gpucb_learner = GPUCB_Learner(n_arms = n_bids, arms = bids)


  for t in tqdm(range(T)):
    pulled_arm = gpts_learner.pull_arm()
    reward = env.draw_n(bids[pulled_arm],noise_std) * opt - env.draw_cc(bids[pulled_arm],noise_std) # 1 is std
    gpts_learner.update(pulled_arm, reward)


    pulled_arm = gpucb_learner.pull_arm()
    reward = env.draw_n(bids[pulled_arm],noise_std) * opt - env.draw_cc(bids[pulled_arm],noise_std)# 1 is std
    gpucb_learner.update(pulled_arm, reward)

  gpts_reward.append(gpts_learner.collected_rewards)
  gpucb_reward.append(gpucb_learner.collected_rewards)

gpts_reward = np.array(gpts_reward)
gpucb_reward = np.array(gpucb_reward)


#print("gpts_num_click_per_experiments",gpts_num_click_per_experiments)
#print("gpts_cum_cost_per_experiments",gpts_cum_cost_per_experiments)

#print("gpucb_num_click_per_experiments",gpucb_num_click_per_experiments)
#print("gpucb_cum_cost_per_experiments",gpucb_cum_cost_per_experiments)

fig, axs = plt.subplots(2,2,figsize=(24,12))

opt_reward = opt * env_array[0].n(optimal_bid) - env_array[0].cc(optimal_bid)

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(gpts_reward, axis = 0)), 'r')
axs[0][0].plot(np.cumsum(np.mean(gpucb_reward, axis = 0)), 'm')


#We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(gpts_reward, axis = 0)), 'b')   
axs[0][0].plot(np.cumsum(np.std(gpucb_reward, axis = 0)), 'c')

axs[0][0].plot(np.cumsum(np.mean(opt_reward - gpts_reward, axis = 0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(opt_reward - gpucb_reward, axis = 0)), 'y')

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
axs[1][0].plot(np.mean(opt_reward - gpts_reward, axis = 0), 'g')
axs[1][0].plot(np.mean(opt_reward - gpucb_reward, axis = 0), 'y')
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
print(gpts_reward)
print(gpucb_reward)
print(opt_reward)