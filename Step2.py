from Classes.learners import GPTS_Learner, GPUCB_Learner
from Classes.enviroment import Environment
from Classes.clairvoyant import clairvoyant

import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

n_prices = 5
n_bids = 100
cost_of_product = 180
price = 100

bids = np.linspace(0.0, 1.0, n_bids)
prices = price * np.array([2, 2.5, 3, 3.5, 4])
margins = np.array([prices[i] - cost_of_product for i in range(n_prices)])
classes = np.array([0, 1, 2])
#                            C1    C2    C3
conversion_rate = np.array([[0.38, 0.41, 0.67],  # 1*price
                            [0.22, 0.24, 0.56],  # 2*price
                            [0.15, 0.19, 0.44],  # 3*price
                            [0.12, 0.09, 0.36],  # 4*price
                            [0.06, 0.05, 0.29]  # 5*price
                            ])

env_array = []
for c in classes:
    env_array.append(Environment(n_prices, conversion_rate[:, c], c))

# EXPERIMENT BEGIN
T = 365

n_experiments = 25
n_noise_std = 2
cc_noise_std = 3

gpts_reward = []
gpucb_reward = []

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

for e in range(n_experiments):
    print(e)
    env = env_array[0]

    n_gpts_learner = GPTS_Learner(n_arms=n_bids, arms=bids)
    cc_gpts_learner = GPTS_Learner(n_arms=n_bids, arms=bids)

    n_gpucb_learner = GPUCB_Learner(n_arms=n_bids, arms=bids)
    cc_gpucb_learner = GPUCB_Learner(n_arms=n_bids, arms=bids)

    gpts_collected_rewards = np.array([])
    gpucb_collected_rewards = np.array([])

    for t in tqdm(range(T)):
        # gpts
        # pull the arm
        sampled_n = np.random.normal(n_gpts_learner.means, n_gpts_learner.sigmas)
        sampled_cc = np.random.normal(cc_gpts_learner.means, cc_gpts_learner.sigmas)
        sampled_reward = sampled_n * opt - sampled_cc
        pulled_arm = np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])

        # play the arm
        n = max(0, env.draw_n(bids[pulled_arm], n_noise_std))
        cc = max(0, env.draw_cc(bids[pulled_arm], cc_noise_std))
        reward = n * opt - cc

        # update learners
        gpts_collected_rewards = np.append(gpts_collected_rewards, reward)
        n_gpts_learner.update(pulled_arm, n)
        cc_gpts_learner.update(pulled_arm, cc)

        # gpucb
        # pull the arm
        sampled_n = n_gpucb_learner.means + n_gpucb_learner.confidence
        sampled_cc = cc_gpucb_learner.means - cc_gpucb_learner.confidence
        sampled_reward = sampled_n * opt - sampled_cc
        pulled_arm = np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])

        # play the arm
        n = max(0, env.draw_n(bids[pulled_arm], n_noise_std))
        cc = max(0, env.draw_cc(bids[pulled_arm], cc_noise_std))
        reward = n * opt - cc

        # update learners
        gpucb_collected_rewards = np.append(gpucb_collected_rewards, reward)
        n_gpucb_learner.update(pulled_arm, n)
        cc_gpucb_learner.update(pulled_arm, cc)

    gpts_reward.append(gpts_collected_rewards)
    gpucb_reward.append(gpucb_collected_rewards)

gpts_reward = np.array(gpts_reward)
gpts_regret = np.array(opt_reward - gpts_reward)
gpts_cum_reward = np.cumsum(gpts_reward, axis=1)
gpts_cum_regret = np.cumsum(gpts_regret, axis=1)

gpucb_reward = np.array(gpucb_reward)
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
