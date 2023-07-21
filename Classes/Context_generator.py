import math
import numpy as np
import pandas as pd


class Context_generator():
  def __init__(self, margins):
    self.margins = margins
    d = {'feature1': [], 'feature2': [], 'conv': [], 'n': [], 'cc': [], 'price': [], 'bid': []}
    self.dataset = pd.DataFrame(data=d)

  def update_dataset(self, feature1, feature2, conv, n, cc, price, bid):
    self.dataset.loc[len(self.dataset)] = [feature1, feature2, conv, n, cc, price, bid]

  def select_context(self):
    confidence = 0.98
    reward = self.get_context_reward(self.dataset)
    reward_male = 0
    if len(self.dataset[(self.dataset['feature1'] == 0)]) > 0:
      reward_male = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 0)])
    reward_female = 0
    if len(self.dataset[(self.dataset['feature1'] == 1)]) > 0:
      reward_female = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 1)])
    p_male = len(self.dataset[(self.dataset['feature1'] == 0)]) / len(self.dataset)
    p_male -= math.sqrt(-(math.log(confidence)/(2*len(self.dataset)))) * p_male
    p_female = len(self.dataset[(self.dataset['feature1'] == 1)]) / len(self.dataset)
    p_female -= math.sqrt(-(math.log(confidence) / (2 * len(self.dataset)))) * p_female
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
      p_male_amateur = len(self.dataset[(self.dataset['feature1'] == 0) & (
              self.dataset['feature2'] == 0)]) / len(self.dataset)
      p_male_amateur -= math.sqrt(-(math.log(confidence) / (2 * len(self.dataset)))) * p_male_amateur
      p_male_pro = len(self.dataset[(self.dataset['feature1'] == 0) & (
              self.dataset['feature2'] == 1)]) / len(self.dataset)
      p_male_pro -= math.sqrt(-(math.log(confidence) / (2 * len(self.dataset)))) * p_male_pro
      p_female_amateur = len(self.dataset[(self.dataset['feature1'] == 1) & (
              self.dataset['feature2'] == 0)]) / len(self.dataset)
      p_female_amateur -= math.sqrt(-(math.log(confidence) / (2 * len(self.dataset)))) * p_female_amateur
      p_female_pro = len(self.dataset[(self.dataset['feature1'] == 1) & (
              self.dataset['feature2'] == 1)]) / len(self.dataset)
      p_female_pro -= math.sqrt(-(math.log(confidence) / (2 * len(self.dataset)))) * p_female_pro
      if p_male_amateur * reward_male_amateur + p_male_pro * reward_male_pro > reward_male:
        if p_female_amateur * reward_female_amateur + p_female_pro * reward_female_pro > reward_female:
          return [3,4,5,6]
        else:
          return [3,4,2]
      if p_female_amateur * reward_female_amateur + p_female_pro * reward_female_pro > reward_female:
        return [5,6,1]
      else:
        return [1,2]
    else:
      return [0]

  def get_context_reward(self, dataset):
    confidence = 0.95
    prices = dataset['price'].unique().tolist()
    conv_rates = []
    for p in prices:
      conv_rates.append(len(dataset[(dataset['price'] == p) & (dataset['conv'] == 1)]) / len(
        dataset[(dataset['price'] == p)]))
    opt_earning = np.max([conv_rates[i] * self.margins[i] for i in range(len(prices))])
    opt_earning -= math.sqrt(-(math.log(confidence) / (2 * len(dataset)))) * opt_earning

    bids = dataset['bid'].unique().tolist()
    n_clicks = []
    cum_costs = []
    for bid in bids:
      n_clicks.append(sum(dataset[(dataset['bid'] == bid)]['n'].to_list()) / len(dataset[(
              dataset['bid'] == bid)]))
      cum_costs.append(sum(dataset[(dataset['bid'] == bid)]['cc'].to_list()) / len(dataset[(
              dataset['bid'] == bid)]))

    reward = np.max([(n_clicks[i] * opt_earning - cum_costs[i]) for i in range(len(bids))])
    reward -= math.sqrt(-(math.log(confidence) / (2 * len(dataset)))) * reward
    return reward
