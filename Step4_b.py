from Classes.learners import TS_Learner, GPTS_Learner, GPUCB_Learner
from Classes.enviroment_step4 import Environment4
from Classes.clairvoyant import clairvoyant4
from Classes.Context import Context
from Classes.Context_generator import Context_generator

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

n_prices = 5
n_bids = 100
cost_of_product = 150
price = 100

bids = np.linspace(0.0, 1.0, n_bids)
prices = price * np.array([2, 2.5, 3, 3.5, 4])
margins = np.array([prices[i] - cost_of_product for i in range(n_prices)])
classes = np.array([0, 1, 2, 3])
#                            C1    C2    C3    C4
conversion_rate = np.array([[0.05, 0.18, 0.28, 0.28],  # 1*price
                            [0.35, 0.20, 0.16, 0.16],  # 2*price
                            [0.15, 0.38, 0.12, 0.12],  # 3*price
                            [0.10, 0.22, 0.10, 0.10],  # 4*price
                            [0.13, 0.15, 0.06, 0.06]  # 5*price
                            ])

env_dict = {
    "NN": Environment4(n_prices, conversion_rate, [None, None]),
    "N0": Environment4(n_prices, conversion_rate, [None, 0]),
    "N1": Environment4(n_prices, conversion_rate, [None, 1]),
    "0N": Environment4(n_prices, conversion_rate, [0, None]),
    "1N": Environment4(n_prices, conversion_rate, [1, None]),
    "00": Environment4(n_prices, conversion_rate, [0, 0]),
    "01": Environment4(n_prices, conversion_rate, [0, 1]),
    "10": Environment4(n_prices, conversion_rate, [1, 0]),
    "11": Environment4(n_prices, conversion_rate, [1, 1])
}

env_keys = ["00", "01", "10", "11"]

opt_price_indexes, opt_bid_indexs, opt_rewards = clairvoyant4(env_keys, bids, prices, margins,
                                                              conversion_rate, env_dict)
print(opt_price_indexes)
print(opt_bid_indexs)
print(opt_rewards)
opt_reward = sum(opt_rewards)
print(opt_reward)

# EXPERIMENT BEGIN FOR ESTIMATING THE OPTIMAL PRICE
T = 365

n_experiments = 25
n_noise_std = 1.5
cc_noise_std = 1

n_context, cc_context = 0, 0
probabilities = [0.5, 0.3]
ts_rewards_per_experiments = [[] for i in range(n_experiments)]
gpts_reward = []
gpucb_reward = []

for e in tqdm(range(n_experiments)):
    contexts_dict = {
        "NN": Context([None, None], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
        "N0": Context([None, 0], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
        "N1": Context([None, 1], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
        "0N": Context([0, None], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
        "1N": Context([1, None], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
        "00": Context([0, 0], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
        "01": Context([0, 1], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
        "10": Context([1, 0], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids)),
        "11": Context([1, 1], TS_Learner(n_arms=n_prices), GPTS_Learner(n_arms=n_bids, arms=bids),
                      GPUCB_Learner(n_arms=n_bids, arms=bids), TS_Learner(n_arms=n_prices),
                      GPTS_Learner(n_arms=n_bids, arms=bids), GPUCB_Learner(n_arms=n_bids, arms=bids))
    }

    context_generator_gpts = Context_generator(margins)
    context_generator_gpucb = Context_generator(margins)
    contexts_gpts = [contexts_dict["NN"]]
    contexts_gpucb = [contexts_dict["NN"]]
    gpts_collected_rewards = np.array([])
    gpucb_collected_rewards = np.array([])
    for t in range(0, T):
        # if t % 14 == 0 and t != 0:  # context generation
        #     contexts_gpts = []
        #     contexts_gpucb = []
        #     print(context_generator_gpts.select_context())
        #     for features in context_generator_gpts.select_context():
        #         contexts_gpts.append(contexts_dict[features])
        #     print(context_generator_gpucb.select_context())
        #     for features in context_generator_gpucb.select_context():
        #         contexts_gpucb.append(contexts_dict[features])

        gpts_reward_iteration = 0
        gpucb_reward_iteration = 0

        # gpts
        # pull the arm
        pulled_arms_per_context = {}  # { [feature1, feature2] : [pulled_arm_price, pulled_arm_bid] }
        for i in range(len(contexts_gpts)):
            pulled_arm_price = (contexts_gpts[i].ts_learner_gpts.pull_arm(margins))
            sampled_n = np.random.normal(contexts_gpts[i].n_gpts_learner.means,
                                         contexts_gpts[i].n_gpts_learner.sigmas)
            sampled_cc = np.random.normal(contexts_gpts[i].cc_gpts_learner.means,
                                          contexts_gpts[i].cc_gpts_learner.sigmas)
            sampled_conv_rate = np.random.beta(contexts_gpts[i].ts_learner_gpts.beta_parameters[pulled_arm_price, 0],
                                               contexts_gpts[i].ts_learner_gpts.beta_parameters[pulled_arm_price, 1])
            sampled_reward = sampled_n * sampled_conv_rate * margins[pulled_arm_price] - sampled_cc
            pulled_arm_bid = np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])
            pulled_arms_per_context[contexts_gpts[i].get_features()] = [pulled_arm_price, pulled_arm_bid]

        # play the arm
        for features_string in pulled_arms_per_context.keys():
            features = []
            for feature in features_string:
                if feature == "N":
                    features.append(None)
                elif feature == "0":
                    features.append(0)
                elif feature == "1":
                    features.append(1)

            n = int(max(0, env_dict[features_string].draw_n(bids[pulled_arms_per_context[features_string][1]],
                                                            n_noise_std)))
            cc = max(0, env_dict[features_string].draw_cc(bids[pulled_arms_per_context[features_string][1]],
                                                          cc_noise_std))
            reward = [0, 0, 0]  # conversions, failures, reward
            users_data = {  # conversions and clicks for each user type
                "00": [0, 0],
                "01": [0, 0],
                "10": [0, 0],
                "11": [0, 0]
            }

            for user in range(n):
                features_user = features.copy()
                if features_user[0] is None:
                    features_user[0] = np.random.binomial(1, 0.5)
                if features_user[1] is None:
                    features_user[1] = np.random.binomial(1, 0.5)
                features_user_string = ""
                for feature in features_user:
                    if feature is None:
                        features_user_string += "N"
                    elif feature == 0:
                        features_user_string += "0"
                    elif feature == 1:
                        features_user_string += "1"
                conversion = env_dict[features_user_string].round(pulled_arms_per_context[features_string][0],
                                                                  features_user)
                reward[0] += conversion
                users_data[features_user_string][0] += conversion
                users_data[features_user_string][1] += 1
            reward[1] = n - reward[0]
            reward[2] = reward[0] * margins[pulled_arms_per_context[features_string][0]] - cc

            # update learners
            for features_ in users_data.keys():
                features_int = []
                for feature in features_:
                    if feature == "N":
                        features_int.append(None)
                    elif feature == "0":
                        features_int.append(0)
                    elif feature == "1":
                        features_int.append(1)
                if users_data[features_][0] != 0:
                    context_generator_gpts.update_dataset(features_int[0], features_int[1],
                                                          pulled_arms_per_context[features_string][0],
                                                          pulled_arms_per_context[features_string][1],
                                                          users_data[features_][0], users_data[features_][1], cc)
            contexts_dict[features_string].ts_learner_gpts.update(pulled_arms_per_context[features_string][0], reward)
            contexts_dict[features_string].n_gpts_learner.update(pulled_arms_per_context[features_string][1], n)
            contexts_dict[features_string].cc_gpts_learner.update(pulled_arms_per_context[features_string][1], cc)
            gpts_reward_iteration += reward[2]

        # gpucb
        # pull the arm
        pulled_arms_per_context = {}  # { [feature1, feature2] : [pulled_arm_price, pulled_arm_bid] }
        for i in range(len(contexts_gpucb)):
            pulled_arm_price = (contexts_gpucb[i].ts_learner_gpucb.pull_arm(margins))
            sampled_n = np.random.normal(contexts_gpucb[i].n_gpucb_learner.means,
                                         contexts_gpucb[i].n_gpucb_learner.sigmas)
            sampled_cc = np.random.normal(contexts_gpucb[i].cc_gpucb_learner.means,
                                          contexts_gpucb[i].cc_gpucb_learner.sigmas)
            sampled_conv_rate = np.random.beta(
                contexts_gpucb[i].ts_learner_gpucb.beta_parameters[pulled_arm_price, 0],
                contexts_gpucb[i].ts_learner_gpucb.beta_parameters[pulled_arm_price, 1])
            sampled_reward = sampled_n * sampled_conv_rate * margins[pulled_arm_price] - sampled_cc
            pulled_arm_bid = np.random.choice(np.where(sampled_reward == sampled_reward.max())[0])
            pulled_arms_per_context[contexts_gpucb[i].get_features()] = [pulled_arm_price, pulled_arm_bid]

        # play the arm
        for features_string in pulled_arms_per_context.keys():
            features = []
            for feature in features_string:
                if feature == "N":
                    features.append(None)
                elif feature == "0":
                    features.append(0)
                elif feature == "1":
                    features.append(1)

            n = int(max(0, env_dict[features_string].draw_n(bids[pulled_arms_per_context[features_string][1]],
                                                            n_noise_std)))
            cc = max(0, env_dict[features_string].draw_cc(bids[pulled_arms_per_context[features_string][1]],
                                                          cc_noise_std))
            reward = [0, 0, 0]  # conversions, failures, reward
            users_data = {  # conversions and clicks for each user type
                "00": [0, 0],
                "01": [0, 0],
                "10": [0, 0],
                "11": [0, 0]
            }

            for user in range(n):
                features_user = features.copy()
                if features_user[0] is None:
                    features_user[0] = np.random.binomial(1, 0.5)
                if features_user[1] is None:
                    features_user[1] = np.random.binomial(1, 0.5)
                features_user_string = ""
                for feature in features_user:
                    if feature is None:
                        features_user_string += "N"
                    elif feature == 0:
                        features_user_string += "0"
                    elif feature == 1:
                        features_user_string += "1"
                conversion = env_dict[features_user_string].round(pulled_arms_per_context[features_string][0],
                                                                  features_user)
                reward[0] += conversion
                users_data[features_user_string][0] += conversion
                users_data[features_user_string][1] += 1
            reward[1] = n - reward[0]
            reward[2] = reward[0] * margins[pulled_arms_per_context[features_string][0]] - cc

            # update learners
            for features_ in users_data.keys():
                features_int = []
                for feature in features_:
                    if feature == "N":
                        features_int.append(None)
                    elif feature == "0":
                        features_int.append(0)
                    elif feature == "1":
                        features_int.append(1)
                if users_data[features_][0] != 0:
                    context_generator_gpucb.update_dataset(features_int[0], features_int[1],
                                                           pulled_arms_per_context[features_string][0],
                                                           pulled_arms_per_context[features_string][1],
                                                           users_data[features_][0], users_data[features_][1], cc)
            contexts_dict[features_string].ts_learner_gpucb.update(pulled_arms_per_context[features_string][0],
                                                                   reward)
            contexts_dict[features_string].n_gpucb_learner.update(pulled_arms_per_context[features_string][1], n)
            contexts_dict[features_string].cc_gpucb_learner.update(pulled_arms_per_context[features_string][1], cc)
            gpucb_reward_iteration += reward[2]

        gpts_collected_rewards = np.append(gpts_collected_rewards, gpts_reward_iteration)
        gpucb_collected_rewards = np.append(gpucb_collected_rewards, gpucb_reward_iteration)

    gpts_reward.append(gpts_collected_rewards)
    gpucb_reward.append(gpucb_collected_rewards)

gpts_rewards = np.array(gpts_reward)
gpucb_rewards = np.array(gpucb_reward)

# total regret is the sum of all regrets
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
