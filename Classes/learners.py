import numpy as np

class Learner():
  def __init__(self, n_arms):
    self.n_arms = n_arms
    self.t = 0
    self.reward_per_arm = x = [[] for i in range(n_arms)]
    self.collected_rewards = np.array([])

  def update_observations(self, pulled_arm, reward): #update the observation list once the reward is returned
    self.reward_per_arm[pulled_arm].append(reward)
    self.collected_rewards = np.append(self.collected_rewards, reward)


class TS_Learner(Learner):
  def __init__(self, n_arms):
    super().__init__(n_arms)
    self.beta_parameters = np.ones((n_arms,2))

  def pull_arm(self):
    idx = np.argmax(np.random.beta(self.beta_parameters[:,0], self.beta_parameters[:,1]))
    return idx

  def update(self, pulled_arm, reward):
    self.t +=1
    self.update_observations(pulled_arm, reward)
    self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
    self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

class UCB1_Learner(Learner):
  def __init__(self, n_arms):
    super().__init__(n_arms)
    self.emprical_means = np.zeros(n_arms)
    self.confidence = np.array([np.inf]*n_arms)

  def pull_arm(self):
    upper_conf = self.emprical_means + self.confidence
    return np.random.choice(np.where(upper_conf == upper_conf.max())[0])

  def update(self, pulled_arm, reward):
    self.t += 1
    self.emprical_means[pulled_arm] = (self.emprical_means[pulled_arm]*(self.t-1)+reward)/self.t
    for a in range(self.n_arms):
      n_samples = len(self.reward_per_arm[a])
      self.confidence[a] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf
    self.update_observations(pulled_arm, reward)
    