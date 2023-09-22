import math
import numpy as np
import pandas as pd


class Context_generator():
    def __init__(self, margins):
        self.margins = margins
        d = {'feature1': [], 'feature2': [], 'price': [], 'bid': [], 'conv': [], 'n': [], 'cc': []}
        self.dataset = pd.DataFrame(data=d)

    def update_dataset(self, feature1, feature2, price, bid, conv, n, cc):
        self.dataset.loc[len(self.dataset)] = [feature1, feature2, price, bid, conv, n, cc]

    def select_context(self):
        confidence = 0.98
        reward = self.get_context_reward(self.dataset)
        reward_male = 0
        if len(self.dataset[(self.dataset['feature1'] == 0)]) > 0:
            reward_male = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 0)])
        reward_female = 0
        if len(self.dataset[(self.dataset['feature1'] == 1)]) > 0:
            reward_female = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 1)])
        p_male = sum(self.dataset[(self.dataset['feature1'] == 0)]['n']) / sum(self.dataset['n'])
        p_male -= math.sqrt(-(math.log(confidence) / (2 * sum(self.dataset[(self.dataset['feature1'] == 0)]['n']))))
        p_female = sum(self.dataset[(self.dataset['feature1'] == 1)]['n']) / sum(self.dataset['n'])
        p_female -= math.sqrt(-(math.log(confidence) / (2 * sum(self.dataset[(self.dataset['feature1'] == 1)]['n']))))
        if p_male * reward_male + p_female * reward_female > reward:
            reward_male_amateur = 0
            if len(self.dataset[(self.dataset['feature1'] == 0) & (self.dataset['feature2'] == 0)]) > 0:
                reward_male_amateur = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 0) &
                                                                           (self.dataset['feature2'] == 0)])
            reward_male_pro = 0
            if len(self.dataset[(self.dataset['feature1'] == 0) & (self.dataset['feature2'] == 1)]) > 0:
                reward_male_pro = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 0) &
                                                                       (self.dataset['feature2'] == 1)])
            reward_female_amateur = 0
            if len(self.dataset[(self.dataset['feature1'] == 1) & (self.dataset['feature2'] == 0)]) > 0:
                reward_female_amateur = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 1) &
                                                                             (self.dataset['feature2'] == 0)])
            reward_female_pro = 0
            if len(self.dataset[(self.dataset['feature1'] == 1) & (self.dataset['feature2'] == 1)]) > 0:
                reward_female_pro = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 1) &
                                                                         (self.dataset['feature2'] == 1)])
            # print("aaa")
            # print(reward_male_pro)
            # print(reward_female_pro)
            # print(reward_female_amateur)
            # print(reward_male_amateur)
            p_male_amateur = sum(self.dataset[(self.dataset['feature1'] == 0) & (
                    self.dataset['feature2'] == 0)]['n']) / sum(self.dataset['n'])
            p_male_amateur -= math.sqrt(-(math.log(confidence) / (2 * sum(self.dataset[(self.dataset[
                                                                                            'feature1'] == 0) & (
                                                                                               self.dataset[
                                                                                                   'feature2'] == 0)][
                                                                              'n']))))
            p_male_pro = sum(self.dataset[(self.dataset['feature1'] == 0) & (
                    self.dataset['feature2'] == 1)]['n']) / sum(self.dataset['n'])
            p_male_pro -= math.sqrt(-(math.log(confidence) / (2 * sum(self.dataset[(self.dataset['feature1'] == 0) & (
                    self.dataset['feature2'] == 1)]['n']))))
            p_female_amateur = sum(self.dataset[(self.dataset['feature1'] == 1) & (
                    self.dataset['feature2'] == 0)]['n']) / sum(self.dataset['n'])
            p_female_amateur -= math.sqrt(-(math.log(confidence) / (2 * sum(self.dataset[(self.dataset[
                                                                                              'feature1'] == 1) & (
                                                                                                 self.dataset[
                                                                                                     'feature2'] == 0)][
                                                                                'n']))))
            p_female_pro = sum(self.dataset[(self.dataset['feature1'] == 1) & (
                    self.dataset['feature2'] == 1)]['n']) / sum(self.dataset['n'])
            p_female_pro -= math.sqrt(-(math.log(confidence) / (2 * sum(self.dataset[(self.dataset['feature1'] == 1) & (
                    self.dataset['feature2'] == 1)]['n']))))
            if p_male_amateur * reward_male_amateur + p_male_pro * reward_male_pro > p_male * reward_male:
                if p_female_amateur * reward_female_amateur + p_female_pro * reward_female_pro > p_female * reward_female:
                    return ["00", "01", "10", "11"]
                else:
                    return ["00", "01", "1N"]
            if p_female_amateur * reward_female_amateur + p_female_pro * reward_female_pro > p_female * reward_female:
                return ["10", "11", "0N"]
            else:
                return ["0N", "1N"]
        else:
            return ["NN"]

    def get_context_reward(self, dataset):
        confidence = 0.98
        prices = dataset['price'].unique().tolist()
        conv_rates = []
        for p in prices:
            conv_rate_estimated = sum(dataset[(dataset['price'] == p)]['conv']) / (len(dataset[(dataset['price'
                                                                                                ] == p)]))
            conv_rate_estimated -= math.sqrt(-(math.log(confidence) / (2 * len(dataset[(dataset['price'] == p)]))))
            conv_rates.append(conv_rate_estimated)
        opt_earning = np.max([conv_rates[i] * self.margins[i] for i in range(len(prices))])

        bids = dataset['bid'].unique().tolist()
        n_clicks = np.array([])
        cum_costs = np.array([])
        for bid in bids:
            n_clicks_estimated = sum(dataset[(dataset['bid'] == bid)]['n'].to_list()) / len(dataset[(
                    dataset['bid'] == bid)])
            n_clicks_estimated -= 1.96 * np.std(dataset[(dataset['bid'] == bid)]['n']) / math.sqrt(len(dataset[(
                    dataset['bid'] == bid)]))
            n_clicks = np.append(n_clicks, n_clicks_estimated)
            cum_costs_estimated = sum(dataset[(dataset['bid'] == bid)]['cc'].to_list()) / len(dataset[(
                    dataset['bid'] == bid)])
            cum_costs_estimated -= 1.96 * np.std(dataset[(dataset['bid'] == bid)]['cc']) / math.sqrt(len(dataset[(
                    dataset['bid'] == bid)]))
            cum_costs = np.append(cum_costs, cum_costs_estimated)

        reward = np.max([((n_clicks[i] * opt_earning - cum_costs[i]) / n_clicks[i]) for i in range(len(bids))])
        return reward
