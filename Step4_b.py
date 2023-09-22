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
cost_of_product = 180
price = 100

bids = np.linspace(0.0, 1.0, n_bids)
prices = price * np.array([2, 2.5, 3, 3.5, 4])
margins = np.array([prices[i] - cost_of_product for i in range(n_prices)])
classes = np.array([0, 1, 2, 3])
#                            C1    C2    C3    C4
conversion_rate = np.array([[0.38, 0.41, 0.57, 0.57],  # 1*price
                            [0.22, 0.24, 0.46, 0.46],  # 2*price
                            [0.15, 0.19, 0.31, 0.31],  # 3*price
                            [0.12, 0.09, 0.23, 0.23],  # 4*price
                            [0.06, 0.05, 0.19, 0.19]  # 5*price
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
T = 200

n_experiments = 10
n_noise_std = 2
cc_noise_std = 3

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
        for features_context in pulled_arms_per_context.keys():
            n = {
                "00": 0,
                "01": 0,
                "10": 0,
                "11": 0
            }
            cc = {
                "00": 0,
                "01": 0,
                "10": 0,
                "11": 0
            }

            n["00"] = int(max(0, env_dict["00"].draw_n(bids[pulled_arms_per_context[
                features_context][1]], n_noise_std)))
            n["01"] = int(max(0, env_dict["01"].draw_n(bids[pulled_arms_per_context[
                features_context][1]], n_noise_std)))
            n["10"] = int(max(0, env_dict["10"].draw_n(bids[pulled_arms_per_context[
                features_context][1]], n_noise_std)))
            n["11"] = int(max(0, env_dict["11"].draw_n(bids[pulled_arms_per_context[
                features_context][1]], n_noise_std)))

            cc["00"] = max(0, env_dict["00"].draw_cc(bids[pulled_arms_per_context[
                features_context][1]], cc_noise_std))
            cc["01"] = max(0, env_dict["01"].draw_cc(bids[pulled_arms_per_context[
                features_context][1]], cc_noise_std))
            cc["10"] = max(0, env_dict["10"].draw_cc(bids[pulled_arms_per_context[
                features_context][1]], cc_noise_std))
            cc["11"] = max(0, env_dict["11"].draw_cc(bids[pulled_arms_per_context[
                features_context][1]], cc_noise_std))
            users_data = {  # conversions and clicks for each user type
                "00": [0, n["00"], cc["00"]],
                "01": [0, n["01"], cc["01"]],
                "10": [0, n["10"], cc["10"]],
                "11": [0, n["11"], cc["11"]]
            }
            reward = [0, 0, 0]  # conversions, failures, reward
            n_obs = 0
            cc_obs = 0
            if features_context == "NN":
                n_obs = sum([n[f] for f in users_data.keys()])
                cc_obs = sum([cc[f] for f in users_data.keys()])
                reward[1] = n_obs
                for user in users_data.keys():
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = (reward[0] * margins[pulled_arms_per_context[features_context][0]]) - cc_obs

            elif features_context == "00" or features_context == "01" or features_context == "10" or \
                    features_context == "11":
                convs = 0
                for i in range(n[features_context]):
                    convs += env_dict[features_context].round(pulled_arms_per_context[features_context][
                                                                  0], features_context)
                reward[0] += convs
                reward[1] = n[features_context] - reward[0]
                users_data[features_context][0] = reward[0]
                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            elif features_context == "0N":
                n_obs = sum([n["00"], n["01"]])
                cc_obs = sum([cc["00"], cc["01"]])
                keys_ = ["00", "01"]
                reward[1] = n_obs
                for user in keys_:
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            elif features_context == "1N":
                n_obs = sum([n["10"], n["11"]])
                cc_obs = sum([cc["10"], cc["11"]])
                keys_ = ["10", "11"]
                reward[1] = n_obs
                for user in keys_:
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            elif features_context == "N0":
                n_obs = sum([n["00"], n["10"]])
                cc_obs = sum([cc["00"], cc["10"]])
                keys_ = ["00", "10"]
                reward[1] = n_obs
                for user in keys_:
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            elif features_context == "N1":
                n_obs = sum([n["10"], n["11"]])
                cc_obs = sum([cc["10"], cc["11"]])
                keys_ = ["10", "11"]
                reward[1] = n_obs
                for user in keys_:
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            # update learners
            for feature_ in users_data.keys():
                features_int = []
                for feature in feature_:
                    if feature == "N":
                        features_int.append(None)
                    elif feature == "0":
                        features_int.append(0)
                    elif feature == "1":
                        features_int.append(1)
                if users_data[feature_][0] != 0:
                    context_generator_gpts.update_dataset(features_int[0], features_int[1],
                                                          pulled_arms_per_context[features_context][0],
                                                          pulled_arms_per_context[features_context][1],
                                                          users_data[feature_][0], users_data[feature_][1],
                                                          users_data[feature_][2])

            contexts_dict[features_context].ts_learner_gpts.update(pulled_arms_per_context[features_context][0],
                                                                   reward)
            contexts_dict[features_context].n_gpts_learner.update(pulled_arms_per_context[features_context][1],
                                                                  n_obs)
            contexts_dict[features_context].cc_gpts_learner.update(pulled_arms_per_context[features_context][1],
                                                                   cc_obs)
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
        for features_context in pulled_arms_per_context.keys():
            n = {
                "00": 0,
                "01": 0,
                "10": 0,
                "11": 0
            }
            cc = {
                "00": 0,
                "01": 0,
                "10": 0,
                "11": 0
            }

            n["00"] = int(max(0, env_dict["00"].draw_n(bids[pulled_arms_per_context[
                features_context][1]], n_noise_std)))
            n["01"] = int(max(0, env_dict["01"].draw_n(bids[pulled_arms_per_context[
                features_context][1]], n_noise_std)))
            n["10"] = int(max(0, env_dict["10"].draw_n(bids[pulled_arms_per_context[
                features_context][1]], n_noise_std)))
            n["11"] = int(max(0, env_dict["11"].draw_n(bids[pulled_arms_per_context[
                features_context][1]], n_noise_std)))

            cc["00"] = max(0, env_dict["00"].draw_cc(bids[pulled_arms_per_context[
                features_context][1]], cc_noise_std))
            cc["01"] = max(0, env_dict["01"].draw_cc(bids[pulled_arms_per_context[
                features_context][1]], cc_noise_std))
            cc["10"] = max(0, env_dict["10"].draw_cc(bids[pulled_arms_per_context[
                features_context][1]], cc_noise_std))
            cc["11"] = max(0, env_dict["11"].draw_cc(bids[pulled_arms_per_context[
                features_context][1]], cc_noise_std))
            users_data = {  # conversions and clicks for each user type
                "00": [0, n["00"], cc["00"]],
                "01": [0, n["01"], cc["01"]],
                "10": [0, n["10"], cc["10"]],
                "11": [0, n["11"], cc["11"]]
            }
            reward = [0, 0, 0]  # conversions, failures, reward
            n_obs = 0
            cc_obs = 0
            if features_context == "NN":
                n_obs = sum([n[f] for f in users_data.keys()])
                cc_obs = sum([cc[f] for f in users_data.keys()])
                reward[1] = n_obs
                for user in users_data.keys():
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = (reward[0] * margins[pulled_arms_per_context[features_context][0]]) - cc_obs

            elif features_context == "00" or features_context == "01" or features_context == "10" or \
                    features_context == "11":
                convs = 0
                for i in range(n[features_context]):
                    convs += env_dict[features_context].round(pulled_arms_per_context[features_context][
                                                                  0], features_context)
                reward[0] += convs
                reward[1] = n[features_context] - reward[0]
                users_data[features_context][0] = reward[0]
                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            elif features_context == "0N":
                n_obs = sum([n["00"], n["01"]])
                cc_obs = sum([cc["00"], cc["01"]])
                keys_ = ["00", "01"]
                reward[1] = n_obs
                for user in keys_:
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            elif features_context == "1N":
                n_obs = sum([n["10"], n["11"]])
                cc_obs = sum([cc["10"], cc["11"]])
                keys_ = ["10", "11"]
                reward[1] = n_obs
                for user in keys_:
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            elif features_context == "N0":
                n_obs = sum([n["00"], n["10"]])
                cc_obs = sum([cc["00"], cc["10"]])
                keys_ = ["00", "10"]
                reward[1] = n_obs
                for user in keys_:
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            elif features_context == "N1":
                n_obs = sum([n["10"], n["11"]])
                cc_obs = sum([cc["10"], cc["11"]])
                keys_ = ["10", "11"]
                reward[1] = n_obs
                for user in keys_:
                    convs = 0
                    for i in range(n[user]):
                        convs += env_dict[user].round(pulled_arms_per_context[features_context][0], user)
                    reward[0] += convs
                    reward[1] -= convs
                    users_data[user][0] = convs

                reward[2] = reward[0] * margins[pulled_arms_per_context[features_context][0]] - cc_obs

            # update learners
            for feature_ in users_data.keys():
                features_int = []
                for feature in feature_:
                    if feature == "N":
                        features_int.append(None)
                    elif feature == "0":
                        features_int.append(0)
                    elif feature == "1":
                        features_int.append(1)
                if users_data[feature_][0] != 0:
                    context_generator_gpucb.update_dataset(features_int[0], features_int[1],
                                                           pulled_arms_per_context[features_context][0],
                                                           pulled_arms_per_context[features_context][1],
                                                           users_data[feature_][0], users_data[feature_][1],
                                                           users_data[feature_][2])

            contexts_dict[features_context].ts_learner_gpucb.update(pulled_arms_per_context[features_context][0],
                                                                    reward)
            contexts_dict[features_context].n_gpucb_learner.update(pulled_arms_per_context[features_context][1],
                                                                   n_obs)
            contexts_dict[features_context].cc_gpucb_learner.update(pulled_arms_per_context[features_context][1],
                                                                    cc_obs)
            gpucb_reward_iteration += reward[2]

        gpts_collected_rewards = np.append(gpts_collected_rewards, gpts_reward_iteration)
        gpucb_collected_rewards = np.append(gpucb_collected_rewards, gpucb_reward_iteration)

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
print(gpts_reward)
print(gpucb_reward)
print(opt_reward)
