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
prices = price * np.array([2, 2.5, 3, 3.5, 4])
margins = np.array([prices[i] - cost_of_product for i in range(n_prices)])
classes = np.array([0, 1, 2])
#                            C1    C2    C3
conversion_rate = np.array([[0.38, 0.41, 0.57],  # 1*price
                            [0.22, 0.24, 0.46],  # 2*price
                            [0.15, 0.19, 0.31],  # 3*price
                            [0.12, 0.09, 0.23],  # 4*price
                            [0.06, 0.05, 0.19]  # 5*price
                            ])

env_array = []
for c in classes:
    env_array.append(Environment(n_prices, conversion_rate[:, c], c))

opt_index_1 = int(clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)[0][0])
opt_index_2 = int(clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)[0][1])
opt_index_3 = int(clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)[0][2])
# opt = normEarnings[opt_index][0]
# 3 classes
opt1 = conversion_rate[opt_index_1][0] * margins[opt_index_1]
opt2 = conversion_rate[opt_index_2][1] * margins[opt_index_2]
opt3 = conversion_rate[opt_index_3][2] * margins[opt_index_3]

optimal_bid_index = clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)[1][0]
optimal_bid_1 = bids[int(optimal_bid_index)]
optimal_bid_index = clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)[1][1]
optimal_bid_2 = bids[int(optimal_bid_index)]
optimal_bid_index = clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)[1][2]
optimal_bid_3 = bids[int(optimal_bid_index)]
print(optimal_bid_1)
print(optimal_bid_2)
print(optimal_bid_3)
print('\n\n')

# EXPERIMENT BEGIN FOR ESTIMATING THE OPTIMAL PRICE
T = 200

n_experiments = 10
n_noise_std = 2
cc_noise_std = 3

ts_rewards_per_experiments = [[], [], []]
gpts_rewards = [[], [], []]
gpucb_rewards = [[], [], []]

for e in tqdm(range(n_experiments)):
    ts_learners_gpts = [TS_Learner(n_arms=n_prices) for c in classes]
    n_gpts_learners = [GPTS_Learner(n_arms=n_bids, arms=bids) for c in classes]
    cc_gpts_learners = [GPTS_Learner(n_arms=n_bids, arms=bids) for c in classes]
    ts_learners_gpucb = [TS_Learner(n_arms=n_prices) for c in classes]
    n_gpucb_learners = [GPUCB_Learner(n_arms=n_bids, arms=bids) for c in classes]
    cc_gpucb_learners = [GPUCB_Learner(n_arms=n_bids, arms=bids) for c in classes]

    gpts_collected_rewards = np.array([])
    gpucb_collected_rewards = np.array([])

    for t in range(0, T):
        for i in range(len(classes)):
            # gpts
            # pull the arm
            pulled_arm_price = ts_learners_gpts[i].pull_arm(margins)
            sampled_n = np.random.normal(n_gpts_learners[i].means, n_gpts_learners[i].sigmas)
            sampled_cc = np.random.normal(cc_gpts_learners[i].means, cc_gpts_learners[i].sigmas)
            sampled_conv_rate = np.random.beta(ts_learners_gpts[i].beta_parameters[pulled_arm_price, 0],
                                               ts_learners_gpts[i].beta_parameters[pulled_arm_price, 1])
            sampled_reward = sampled_n * sampled_conv_rate * margins[pulled_arm_price] - sampled_cc
            pulled_arm_bid = np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])

            # play the arm
            n = int(max(0, env_array[i].draw_n(bids[pulled_arm_bid], n_noise_std)))
            cc = max(0, env_array[i].draw_cc(bids[pulled_arm_bid], cc_noise_std))

            reward = [0, 0, 0]  # conversions, failures, reward
            for user in range(n):
                reward[0] += env_array[i].round(pulled_arm_price)
            reward[1] = n - reward[0]
            reward[2] = reward[0] * margins[pulled_arm_price] - cc

            # update learners
            ts_learners_gpts[i].update(pulled_arm_price, reward)
            n_gpts_learners[i].update(pulled_arm_bid, n)
            cc_gpts_learners[i].update(pulled_arm_bid, cc)
            gpts_collected_rewards = np.append(gpts_collected_rewards, reward[2])

            # gpucb
            # pull the arm
            pulled_arm_price = ts_learners_gpucb[i].pull_arm(margins)
            sampled_n = np.random.normal(n_gpucb_learners[i].means, n_gpucb_learners[i].sigmas)
            sampled_cc = np.random.normal(cc_gpucb_learners[i].means, cc_gpucb_learners[i].sigmas)
            sampled_conv_rate = np.random.beta(ts_learners_gpucb[i].beta_parameters[pulled_arm_price, 0],
                                               ts_learners_gpucb[i].beta_parameters[pulled_arm_price, 1])
            sampled_reward = sampled_n * sampled_conv_rate * margins[pulled_arm_price] - sampled_cc
            pulled_arm_bid = np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])

            # play the arm
            n = int(max(0, env_array[i].draw_n(bids[pulled_arm_bid], n_noise_std)))
            cc = max(0, env_array[i].draw_cc(bids[pulled_arm_bid], cc_noise_std))

            reward = [0, 0, 0]  # conversions, failures, reward
            for user in range(n):
                reward[0] += env_array[i].round(pulled_arm_price)
            reward[1] = n - reward[0]
            reward[2] = reward[0] * margins[pulled_arm_price] - cc

            # update learners
            ts_learners_gpucb[i].update(pulled_arm_price, reward)
            n_gpucb_learners[i].update(pulled_arm_bid, n)
            cc_gpucb_learners[i].update(pulled_arm_bid, cc)

    for i in range(len(classes)):
        gpts_rewards[i].append(ts_learners_gpts[i].collected_rewards)
        gpucb_rewards[i].append(ts_learners_gpucb[i].collected_rewards)

for i in range(len(classes)):
    gpts_rewards[i] = np.array(gpts_rewards[i])
    gpucb_rewards[i] = np.array(gpucb_rewards[i])

opt_reward_1 = opt1 * env_array[0].n(optimal_bid_1) - env_array[0].cc(optimal_bid_1)
opt_reward_2 = opt2 * env_array[1].n(optimal_bid_2) - env_array[1].cc(optimal_bid_2)
opt_reward_3 = opt3 * env_array[2].n(optimal_bid_3) - env_array[2].cc(optimal_bid_3)
print(opt_reward_1)
print(opt_reward_2)
print(opt_reward_3)
print(opt_reward_1 + opt_reward_2 + opt_reward_3)
opt_reward = opt_reward_1 + opt_reward_2 + opt_reward_3

gpts_reward = gpts_rewards[0] + gpts_rewards[1] + gpts_rewards[2]
gpucb_reward = gpucb_rewards[0] + gpucb_rewards[1] + gpucb_rewards[2]

gpts_regret = np.array(opt_reward - gpts_reward)
gpts_cum_reward = np.cumsum(gpts_reward, axis=1)
gpts_cum_regret = np.cumsum(gpts_regret, axis=1)

gpucb_regret = np.array(opt_reward - gpucb_reward)
gpucb_cum_reward = np.cumsum(gpucb_reward, axis=1)
gpucb_cum_regret = np.cumsum(gpucb_regret, axis=1)

fig, axs = plt.subplots(2, 2, figsize=(14, 7))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Cumulative reward")
axs[0][0].plot(np.mean(gpts_cum_reward, axis=0), 'r')
axs[0][0].plot(np.mean(gpucb_cum_reward, axis=0), 'm')
axs[0][0].fill_between(range(T), np.mean(gpts_cum_reward, axis=0) - np.std(
    gpts_cum_reward, axis=0), np.mean(gpts_cum_reward, axis=0) + np.std(
    gpts_cum_reward, axis=0), color='r', alpha=0.2)
axs[0][0].fill_between(range(T), np.mean(gpucb_cum_reward, axis=0) - np.std(
    gpucb_cum_reward, axis=0), np.mean(gpucb_cum_reward, axis=0) + np.std(
    gpucb_cum_reward, axis=0), color='m', alpha=0.2)

axs[0][0].legend(["Cumulative Reward GPTS", "Cumulative Reward GPUCB"])
axs[0][0].set_title("Cumulative Reward GPTS vs GPUCB")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Instantaneous Reward")
axs[0][1].plot(np.mean(gpts_reward, axis=0), 'r')
axs[0][1].plot(np.mean(gpucb_reward, axis=0), 'm')
axs[0][1].fill_between(range(T), np.mean(gpts_reward, axis=0) - np.std(gpts_reward, axis=0),
                       np.mean(gpts_reward, axis=0) + np.std(gpts_reward, axis=0), color='r', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(gpucb_reward, axis=0) - np.std(gpucb_reward, axis=0),
                       np.mean(gpucb_reward, axis=0) + np.std(gpucb_reward, axis=0), color='m', alpha=0.2)
axs[0][1].legend(["Reward GPTS", "Reward GPUCB"])
axs[0][1].set_title("Instantaneous Reward GPTS vs GPUCB")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Cumulative regret")
axs[1][0].plot(np.mean(gpts_cum_regret, axis=0), 'g')
axs[1][0].plot(np.mean(gpucb_cum_regret, axis=0), 'y')
axs[1][0].fill_between(range(T), np.mean(gpts_cum_regret, axis=0) - np.std(
    gpts_cum_regret, axis=0), np.mean(gpts_cum_regret, axis=0) + np.std(gpts_cum_regret, axis=0),
                       color='g', alpha=0.2)
axs[1][0].fill_between(range(T), np.mean(gpucb_cum_regret, axis=0) - np.std(
    gpucb_cum_regret, axis=0), np.mean(gpucb_cum_regret, axis=0) + np.std(gpucb_cum_regret, axis=0),
                       color='g', alpha=0.2)

axs[1][0].legend(["Cumulative Regret GPTS", "Cumulative Regret GPUCB"])
axs[1][0].set_title("Cumulative Regret GPTS vs GPUCB")

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Instantaneous regret")
axs[1][1].plot(np.mean(gpts_regret, axis=0), 'g')
axs[1][1].plot(np.mean(gpucb_regret, axis=0), 'y')
axs[1][1].fill_between(range(T), np.mean(gpts_regret, axis=0) - np.std(
    gpts_regret, axis=0), np.mean(gpts_regret, axis=0) + np.std(gpts_regret, axis=0), color='g', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(gpucb_regret, axis=0) - np.std(
    gpucb_regret, axis=0), np.mean(gpucb_regret, axis=0) + np.std(gpucb_regret, axis=0), color='y', alpha=0.2)
axs[1][1].legend(["Regret GPTS", "Regret GPUCB"])
axs[1][1].set_title("Instantaneous Regret GPTS vs GPUCB")

plt.show()
print(gpts_reward)
print(gpucb_reward)
print(opt_reward)
