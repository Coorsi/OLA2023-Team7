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
    reward = self.get_context_reward(self.dataset)
    reward_male = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 0)])
    reward_female = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 1)])
    print(reward)
    print(reward_male)
    print(reward_female)
    if reward_male + reward_female > 1.5 * reward:
      reward_male_amateur = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 0) &
                                                                 (self.dataset['feature2'] == 0)])
      reward_male_pro = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 0) &
                                                             (self.dataset['feature2'] == 1)])
      reward_female_amateur = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 1) &
                                                                 (self.dataset['feature2'] == 0)])
      reward_female_pro = self.get_context_reward(self.dataset[(self.dataset['feature1'] == 1) &
                                                                 (self.dataset['feature2'] == 1)])
      if reward_male_amateur + reward_male_pro > 1.5 * reward_male:
        if reward_female_amateur + reward_female_pro > 1.5 * reward_female:
          return [3,4,5,6]
        else:
          return [3,4,2]
      if reward_female_amateur + reward_female_pro > 1.5 * reward_female:
        return [5,6,1]
      else:
        return [1,2]
    else:
      return [0]

  def get_context_reward(self, dataset):
    prices = dataset['price'].unique().tolist()
    conv_rates = []
    for p in prices:
      conv_rates.append(len(dataset[(dataset['price'] == p) & (dataset['conv'] == 1)]) / len(
        dataset[(dataset['price'] == p)]))
    opt_earning = np.max([conv_rates[i] * self.margins[i] for i in range(len(prices))])

    bids = dataset['bid'].unique().tolist()
    n_clicks = []
    cum_costs = []
    for bid in bids:
      n_clicks.append(sum(dataset[(dataset['bid'] == bid)]['n'].to_list()) / len(dataset[(
              dataset['bid'] == bid)]))
      cum_costs.append(sum(dataset[(dataset['bid'] == bid)]['cc'].to_list()) / len(dataset[(
              dataset['bid'] == bid)]))

    reward = np.max([(n_clicks[i] * opt_earning - cum_costs[i])  for i in range(len(prices))])
    return reward
