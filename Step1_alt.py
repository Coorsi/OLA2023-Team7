from Classes.learners import TS_Learner,UCB1_Learner
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

env_array = []
for c in classes:
    env_array.append(Environment(n_prices, conversion_rate[:, c], c))


# EXPERIMENT BEGIN
T = 365

n_experiments = 200

ts_rewards_per_experiments = []
ucb1_rewards_per_experiments = []

optimal = clairvoyant(classes, bids, prices, margins, conversion_rate, env_array)
opt_index = int(optimal[0][0])
print(opt_index)
optimal_bid_index = optimal[1][0]
optimal_bid = bids[int(optimal_bid_index)]
opt = optimal[2][0]
print(optimal_bid)
print(opt)

for e in tqdm(range(n_experiments)):
    env = env_array[0]
    ts_learner = TS_Learner(n_arms=n_prices)
    ucb1_learner = UCB1_Learner(n_arms=n_prices)
    for t in range(0, T):
        pulled_arm = ts_learner.pull_arm(margins)
        n = int(env.draw_n(optimal_bid, 1))
        cc = env.draw_cc(optimal_bid, 1)
        reward = [0, 0, 0]    # conversions, failures, reward
        for user in range(int(n)):
            reward[0] += env.round(pulled_arm)
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm] - cc
        ts_learner.update(pulled_arm, reward)

        pulled_arm = ucb1_learner.pull_arm(margins)
        reward = [0, 0, 0]  # conversions, failures, reward
        for user in range(int(n)):
            reward[0] += env.round(pulled_arm)
        reward[1] = n - reward[0]
        reward[2] = reward[0] * margins[pulled_arm] - cc
        ucb1_learner.update(pulled_arm, reward)

    ts_rewards_per_experiments.append(ts_learner.collected_rewards)
    ucb1_rewards_per_experiments.append(ucb1_learner.collected_rewards)

# num_arms_pulled = np.array(list(map(lambda x: len(x),ts_learner.reward_per_arm)))
# learned_optimal_price_index = np.argmax(num_arms_pulled)


ts_rewards_per_experiments = np.array(ts_rewards_per_experiments)
ucb1_rewards_per_experiments = np.array(ucb1_rewards_per_experiments)

fig, axs = plt.subplots(2, 2, figsize=(14, 7))

axs[0][0].set_xlabel("t")
axs[0][0].plot(np.cumsum(np.mean(ts_rewards_per_experiments, axis=0)), 'r')
axs[0][0].plot(np.cumsum(np.mean(ucb1_rewards_per_experiments, axis=0)), 'm')

# We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[0][0].plot(np.cumsum(np.std(ts_rewards_per_experiments, axis=0)), 'b')
axs[0][0].plot(np.cumsum(np.std(ucb1_rewards_per_experiments, axis=0)), 'c')

axs[0][0].plot(np.cumsum(np.mean(opt - ts_rewards_per_experiments, axis=0)), 'g')
axs[0][0].plot(np.cumsum(np.mean(opt - ucb1_rewards_per_experiments, axis=0)), 'y')

axs[0][0].legend(["Reward TS", "Reward UCB1", "Std TS", "Std UCB1", "Regret TS", "Regret UCB1"])
axs[0][0].set_title("Cumulative TS vs UCB1")

axs[0][1].set_xlabel("t")
axs[0][1].set_ylabel("Reward")
axs[0][1].plot(np.mean(ts_rewards_per_experiments, axis=0), 'r')
axs[0][1].plot(np.mean(ucb1_rewards_per_experiments, axis=0), 'm')
axs[0][1].fill_between(range(T), np.mean(ts_rewards_per_experiments, axis=0) - np.std(ts_rewards_per_experiments, axis=0),
                 np.mean(ts_rewards_per_experiments, axis=0) + np.std(ts_rewards_per_experiments,
                                                                      axis=0), color='r', alpha=0.2)
axs[0][1].fill_between(range(T), np.mean(ucb1_rewards_per_experiments, axis=0) - np.std(ucb1_rewards_per_experiments, axis=0),
                 np.mean(ucb1_rewards_per_experiments, axis=0) + np.std(ucb1_rewards_per_experiments,
                                                                        axis=0), color='m', alpha=0.2)
axs[0][1].legend(["Reward TS", "Reward UCB1"])
axs[0][1].set_title("Instantaneous Reward TS vs UCB1")

# We plot only the standard deviation of the reward beacuse the standard deviation of the regret is the same
axs[1][0].set_xlabel("t")
axs[1][0].set_ylabel("Std")
axs[1][0].plot(np.std(ts_rewards_per_experiments, axis=0), 'b')
axs[1][0].plot(np.std(ucb1_rewards_per_experiments, axis=0), 'c')
axs[1][0].legend(["Std TS", "Std UCB1"])
axs[1][0].set_title("Instantaneous Std TS vs UCB1")

axs[1][1].set_xlabel("t")
axs[1][1].set_ylabel("Regret")
axs[1][1].plot(np.mean(opt - ts_rewards_per_experiments, axis=0), 'g')
axs[1][1].plot(np.mean(opt - ucb1_rewards_per_experiments, axis=0), 'y')
axs[1][1].fill_between(range(T), np.mean(opt - ts_rewards_per_experiments, axis=0) - np.std(
        opt - ts_rewards_per_experiments, axis=0), np.mean(opt - ts_rewards_per_experiments, axis=0) + np.std(
        opt - ts_rewards_per_experiments, axis=0), color='g', alpha=0.2)
axs[1][1].fill_between(range(T), np.mean(opt - ucb1_rewards_per_experiments, axis=0) - np.std(
        opt - ucb1_rewards_per_experiments, axis=0), np.mean(opt - ucb1_rewards_per_experiments, axis=0) + np.std(
        opt - ucb1_rewards_per_experiments, axis=0), color='y', alpha=0.2)
axs[1][1].legend(["Regret TS", "Regret UCB1"])
axs[1][1].set_title("Instantaneous Regret TS vs UCB1")

plt.show()

