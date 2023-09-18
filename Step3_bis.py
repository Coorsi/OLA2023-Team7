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
prices = price * np.array([1, 2, 3, 4, 5])
margins = np.array([prices[i] - cost_of_product for i in range(n_prices)])
classes = np.array([0, 1, 2])
#                            C1    C2    C3
conversion_rate = np.array([[0.93, 0.95, 0.77],  # 1*price
                            [0.82, 0.84, 0.42],  # 2*price
                            [0.51, 0.64, 0.29],  # 3*price
                            [0.38, 0.50, 0.21],  # 4*price
                            [0.09, 0.18, 0.11]  # 5*price
                            ])

env_array = []
for c in classes:
    env_array.append(Environment(n_prices, conversion_rate[:, c], c))

optimal = clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)
opt_index = int(optimal[0][0])
print(opt_index)
opt = conversion_rate[opt_index, 0] * margins[opt_index]
print(opt)
optimal_bid_index = optimal[1][0]
optimal_bid = bids[int(optimal_bid_index)]
opt_reward = optimal[2][0]
print(optimal_bid)
print(opt_reward)

# EXPERIMENT BEGIN FOR ESTIMATING THE OPTIMAL PRICE
T = 365

n_experiments = 20
n_noise_std = 1.5
cc_noise_std = 1

gpts_reward = []
gpucb_reward = []

for e in tqdm(range(n_experiments)):
    env = env_array[0]
    ts_learner_1 = TS_Learner(n_arms=n_prices)
    ts_learner_2 = TS_Learner(n_arms=n_prices)
    n_gpts_learner = GPTS_Learner(n_arms=n_bids, arms=bids)
    cc_gpts_learner = GPTS_Learner(n_arms=n_bids, arms=bids)
    n_gpucb_learner = GPUCB_Learner(n_arms=n_bids, arms=bids)
    cc_gpucb_learner = GPUCB_Learner(n_arms=n_bids, arms=bids)

    gpts_collected_rewards = np.array([])
    gpucb_collected_rewards = np.array([])

    for t in range(0, T):
        # gpts
        # pull the arm
        pulled_arm_price = ts_learner_1.pull_arm(margins)
        sampled_n = np.random.normal(n_gpts_learner.means, n_gpts_learner.sigmas)
        sampled_cc = np.random.normal(cc_gpts_learner.means, cc_gpts_learner.sigmas)
        sampled_conv_rate = np.random.beta(ts_learner_1.beta_parameters[pulled_arm_price, 0],
                                           ts_learner_1.beta_parameters[pulled_arm_price, 1])
        sampled_reward = sampled_n * sampled_conv_rate * margins[pulled_arm_price] - sampled_cc
        pulled_arm_bid = np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])

        # play the arm
        n = int(max(0, env.draw_n(bids[pulled_arm_bid], n_noise_std)))
        cc = max(0, env.draw_cc(bids[pulled_arm_bid], cc_noise_std))

        reward = [0, 0, 0]  # conversions, failures, reward
        for user in range(n):
            reward[0] += env.round(pulled_arm_price)
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm_price] - cc

        # update learners
        ts_learner_1.update(pulled_arm_price, reward)
        n_gpts_learner.update(pulled_arm_bid, n)
        cc_gpts_learner.update(pulled_arm_bid, cc)
        gpts_collected_rewards = np.append(gpts_collected_rewards, reward[2])

        # gpucb
        # pull the arm
        pulled_arm_price = ts_learner_2.pull_arm(margins)
        sampled_n = np.random.normal(n_gpucb_learner.means, n_gpucb_learner.sigmas)
        sampled_cc = np.random.normal(cc_gpucb_learner.means, cc_gpucb_learner.sigmas)
        sampled_conv_rate = np.random.beta(ts_learner_2.beta_parameters[pulled_arm_price, 0],
                                           ts_learner_2.beta_parameters[pulled_arm_price, 1])
        sampled_reward = sampled_n * sampled_conv_rate * margins[pulled_arm_price] - sampled_cc
        pulled_arm_bid = np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])

        # play the arm
        n = int(max(0, env.draw_n(bids[pulled_arm_bid], n_noise_std)))
        cc = max(0, env.draw_cc(bids[pulled_arm_bid], cc_noise_std))

        reward = [0, 0, 0]  # conversions, failures, reward
        for user in range(n):
            reward[0] += env.round(pulled_arm_price)
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm_price] - cc

        # update learners
        ts_learner_2.update(pulled_arm_price, reward)
        n_gpucb_learner.update(pulled_arm_bid, n)
        cc_gpucb_learner.update(pulled_arm_bid, cc)
        gpucb_collected_rewards = np.append(gpucb_collected_rewards, reward[2])

    gpts_reward.append(gpts_collected_rewards)
    gpucb_reward.append(gpucb_collected_rewards)

gpts_reward = np.array(gpts_reward)
gpucb_reward = np.array(gpucb_reward)

fig, axs = plt.subplots(2, 2, figsize=(24, 12))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(gpts_reward, axis=0)), 'r')
axs[0][0].plot(np.cumsum(np.mean(gpucb_reward, axis=0)), 'm')

# We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(gpts_reward, axis=0)), 'b')
axs[0][0].plot(np.cumsum(np.std(gpucb_reward, axis=0)), 'c')

axs[0][0].plot(np.cumsum(np.mean(opt_reward - gpts_reward, axis=0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(opt_reward - gpucb_reward, axis=0)), 'y')

axs[0][0].legend(["Reward GPTS", "Reward GPUCB", "Std GPTS", "Std GPUCB", "Regret GPTS", "Regret GPUCB"])
axs[0][0].set_title("Cumulative GPTS vs GPUCB")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.mean(gpts_reward, axis=0), 'r')
axs[0][1].plot(np.mean(gpucb_reward, axis=0), 'm')
axs[0][1].legend(["Reward GPTS", "Reward GPUCB"])
axs[0][1].set_title("Instantaneous Reward GPTS vs GPUCB")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Regret")
axs[1][0].plot(np.mean(opt_reward - gpts_reward, axis=0), 'g')
axs[1][0].plot(np.mean(opt_reward - gpucb_reward, axis=0), 'y')
axs[1][0].legend(["Regret GPTS", "Regret GPUCB"])
axs[1][0].set_title("Instantaneous Std GPTS vs GPUCB")

# We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Reward")
axs[1][1].plot(np.std(gpts_reward, axis=0), 'b')
axs[1][1].plot(np.std(gpucb_reward, axis=0), 'c')
axs[1][1].legend(["Std GPTS", "Std GPUCB"])
axs[1][1].set_title("Instantaneous STD GPTS vs GPUCB")

plt.show()
print(gpts_reward)
print(gpucb_reward)
print(opt_reward)
