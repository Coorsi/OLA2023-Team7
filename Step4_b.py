from Classes.learners import TS_Learner, GPTS_Learner, GPUCB_Learner
from Classes.enviroment_step4 import Environment
from Classes.clairvoyant import clairvoyant
from Classes.Context import Context
from Classes.Context_generator import Context_generator

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
classes = np.array([0, 1, 2, 3])
# C1   C2   C3   C4
conversion_rate = np.array([[0.83, 0.96, 0.77, 0.77],  # 1*price
                            [0.77, 0.89, 0.42, 0.42],  # 2*price
                            [0.51, 0.68, 0.29, 0.29],  # 3*price
                            [0.24, 0.54, 0.21, 0.21],  # 4*price
                            [0.09, 0.21, 0.11, 0.11]  # 5*price
                            ])

n1 = lambda x: 5 * (1 - np.exp(-5 * x + 2 * x ** 3))
n2 = lambda x: 9 * (1 - np.exp(-2 * x + 2 * x ** 3))
n3 = lambda x: 5 * (1 - np.exp(-3 * x + 2 * x ** 3))

cc1 = lambda x: 2 * (1 - np.exp(-3 * x + 2 * x ** 2))
cc2 = lambda x: 2 * (1 - np.exp(-2 * x + 2 * x ** 2))
cc3 = lambda x: 2 * (1 - np.exp(-2.5 * x + 2 * x ** 2))

n = [n1, n2, n3, n3]
cc = [cc1, cc2, cc3, cc3]

env_array = []
for c in classes:
    env_array.append(Environment(n_prices, conversion_rate[:, c], n[c], cc[c]))

opt_indexes = []
opts = []
for i in classes:
    opt_indexes.append(int(clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)[0][i]))
# opt = normEarnings[opt_index][0]
for i in classes:
    opts.append(conversion_rate[opt_indexes[i]][i] * margins[opt_indexes[i]])

optimal_bids = []
for i in classes:
    optimal_bid_index = clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)[1][i]
    optimal_bids.append(bids[int(optimal_bid_index)])

print(optimal_bids[0])
print(optimal_bids[1])
print('\n\n')

# EXPERIMENT BEGIN FOR ESTIMATING THE OPTIMAL PRICE
T = 365

n_experiments = 40
noise_std = 0.6

n_context, cc_context = 0, 0
probabilities = [0.5, 0.3]
ts_rewards_per_experiments = [[] for i in range(n_experiments)]
gpts_rewards = np.zeros((n_experiments, T))
gpucb_rewards = np.zeros((n_experiments, T))

for e in tqdm(range(n_experiments)):
    contexts_tot = [Context([None, None], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                            GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                            GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
                    Context([0, None], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                            GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                            GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
                    Context([1, None], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                            GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                            GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
                    Context([0, 0], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                            GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                            GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
                    Context([0, 1], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                            GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                            GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
                    Context([1, 0], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                            GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                            GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
                    Context([1, 1], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                            GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                            GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids))
                    ]
    context_generator = Context_generator(margins)
    contexts = [contexts_tot[0]]
    gpts_reward = np.zeros(T)
    gpucb_reward = np.zeros(T)
    for t in range(0, T):
        if t % 14 == 0 and t != 0:
            contexts = []
            print(context_generator.select_context())
            for i in context_generator.select_context():
                contexts.append(contexts_tot[i])

        features = [np.random.binomial(1, probabilities[0]), np.random.binomial(1, probabilities[1])]
        for i in range(len(contexts)):
            if (contexts[i].features[0] == features[0] or contexts[i].features[
                     0] is None) and (contexts[i].features[1] == features[1] or contexts[i].features[1] is None):
                index = features[1] + (2 * features[0])
                pulled_arm_price = contexts[i].ts_learner.pull_arm()
                reward = env_array[index].round(pulled_arm_price)
                sampled_normEarning = np.random.beta(contexts[i].ts_learner.beta_parameters[pulled_arm_price, 0],
                                                     contexts[i].ts_learner.beta_parameters[pulled_arm_price, 1])
                # print(sampled_normEarning)
                contexts[i].ts_learner.update(pulled_arm_price, reward)

                pulled_arm_bid = contexts[i].gpts_learner.pull_arm()
                drawed_n = env_array[index].draw_n(bids[pulled_arm_bid], noise_std)
                drawed_cc = env_array[index].draw_cc(bids[pulled_arm_bid], noise_std)

                reward_tot = drawed_n * sampled_normEarning - drawed_cc
                contexts[i].gpts_learner.update(pulled_arm_bid, reward_tot)

                context_generator.update_dataset(features[0], features[1], reward, drawed_n, drawed_cc,
                                                 pulled_arm_price, pulled_arm_bid)
                gpts_reward[t] = reward_tot

                pulled_arm_bid = contexts[i].gpucb_learner.pull_arm()
                drawed_n_ucb = env_array[index].draw_n(bids[pulled_arm_bid], noise_std)
                drawed_cc_ucb = env_array[index].draw_cc(bids[pulled_arm_bid], noise_std)
                reward_tot = drawed_n_ucb * sampled_normEarning - drawed_cc_ucb
                contexts[i].gpucb_learner.update(pulled_arm_bid, reward_tot)

                gpucb_reward[t] = reward_tot

    gpts_rewards[e] = gpts_reward
    gpucb_rewards[e] = gpucb_reward


gpts_rewards = np.array(gpts_rewards)
gpucb_rewards = np.array(gpucb_rewards)

opt_rewards = []
for i in range(len(classes)):
    opt_rewards.append(opts[i] * env_array[i].n(optimal_bids[i]) - env_array[i].cc(optimal_bids[i]))

# total regret is the sum of all regrets
opt_reward = np.sum([opt_rewards[i] for i in range(len(opt_rewards))]) / len(classes)
tot_regret_gpts = opt_reward - gpts_rewards
tot_regret_gpucb = opt_reward - gpucb_rewards


# plot
fig, axs = plt.subplots(2, 2, figsize=(24, 12))

axs[0][0].set_xlabel("t")
axs[0][0].set_ylabel("Regret")
axs[0][0].plot(np.cumsum(np.mean(gpts_rewards, axis=0)), 'r')
axs[0][0].plot(np.cumsum(np.mean(gpucb_rewards, axis=0)), 'm')

# We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(gpts_rewards, axis=0)), 'b')
axs[0][0].plot(np.cumsum(np.std(gpucb_rewards, axis=0)), 'c')

axs[0][0].plot(np.cumsum(np.mean(tot_regret_gpts, axis=0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(tot_regret_gpucb, axis=0)), 'y')

axs[0][0].legend(["Reward GPTS", "Reward GPUCB", "Std GPTS", "Std GPUCB", "Regret GPTS", "Regret GPUCB"])
axs[0][0].set_title("Cumulative GPTS vs GPUCB")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.mean(gpts_rewards, axis=0), 'r')
axs[0][1].plot(np.mean(gpucb_rewards, axis=0), 'm')
axs[0][1].legend(["Reward GPTS", "Reward GPUCB"])
axs[0][1].set_title("Instantaneous Reward GPTS vs GPUCB")

axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Regret")
axs[1][0].plot(np.mean(tot_regret_gpts, axis=0), 'g')
axs[1][0].plot(np.mean(tot_regret_gpucb, axis=0), 'y')
axs[1][0].legend(["Regret GPTS", "Regret GPUCB"])
axs[1][0].set_title("Instantaneous Std GPTS vs GPUCB")

# We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Reward")
axs[1][1].plot(np.std(gpts_rewards, axis=0), 'b')
axs[1][1].plot(np.std(gpucb_rewards, axis=0), 'c')
axs[1][1].legend(["Std GPTS", "Std GPUCB"])
axs[1][1].set_title("Instantaneous Reward GPTS vs GPUCB")

plt.show()
# print(gpts_rewards)
# print(gpucb_rewards)
# print(opt_reward)
