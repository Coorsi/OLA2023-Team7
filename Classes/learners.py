import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

  

class Learner():
  def __init__(self, n_arms):
    self.n_arms = n_arms
    self.t = 0
    self.reward_per_arm = [[] for i in range(n_arms)]
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


class SWTS_Learner(TS_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])
    
    def update(self, pulled_arm, reward):
        self.current += 1
        self.update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            cum_rew = np.sum(self.reward_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            self.beta_parameters[arm,0] = cum_rew + 1
            self.beta_parameters[arm,1] = n_samples - cum_rew + 1

class SWUCB_Learner(UCB1_Learner):
    def __init__(self, n_arms, window_size):
        super().__init__(n_arms)
        self.window_size = window_size
        self.pulled_arms = np.array([])
    
    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.pulled_arms = np.append(self.pulled_arms, pulled_arm)
        for arm in range(self.n_arms):
            n_samples = np.sum(self.pulled_arms[-self.window_size:] == arm)
            cum_rew = np.sum(self.reward_per_arm[arm][-n_samples:]) if n_samples > 0 else 0
            self.emprical_means[arm] = cum_rew / n_samples if n_samples > 0 else 0
            self.confidence[arm] = (2*np.log(self.t)/n_samples)**0.5 if n_samples > 0 else np.inf

class GPTS_Learner(Learner):
  def __init__(self, n_arms, arms):
    super().__init__(n_arms)
    self.arms = arms
    self.means = np.zeros(n_arms)
    self.sigmas = np.ones(n_arms)*10
    self.pulled_arms = []
    alpha = 1
    kernel = C(1e1, (1e-7, 1e7)) * RBF(1e1, (1e-10, 1e7))
    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2)
    self.iteration = 0
    self.refitFrequency = 13 #13 bc it divides 364

  def update_observations(self, pulled_arm, reward):
    super().update_observations(pulled_arm, reward)
    self.pulled_arms.append(self.arms[pulled_arm])
  
  def update_model(self):
    self.iteration += 1
    x = np.atleast_2d(self.pulled_arms).T
    y = self.collected_rewards
    if self.iteration % self.refitFrequency == 0:
      self.gp.fit(x,y)
    self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
    self.sigmas = np.maximum(self.sigmas, 1e-2)

  def update(self, pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm, reward)
    self.update_model()

  def pull_arm(self):
    sampled_values = np.random.normal(self.means, self.sigmas)
    return np.argmax(sampled_values)


class GPUCB_Learner(Learner): 
  def __init__(self, n_arms, arms):
    super().__init__(n_arms)
    self.arms = arms
    self.means = np.zeros(n_arms)
    self.sigmas = np.ones(n_arms)*np.inf
    self.pulled_arms = []
    alpha = 1
    kernel = C(1e1, (1e-7, 1e7)) * RBF(1e1, (1e-10, 1e7))
    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2,n_restarts_optimizer=5)
    self.iteration = 0
    self.refitFrequency = 13 #13 bc it divides 364

  def update_observations(self, pulled_arm, reward):
    super().update_observations(pulled_arm, reward)
    self.pulled_arms.append(self.arms[pulled_arm])

  def update_model(self):
    self.iteration += 1
    x = np.atleast_2d(self.pulled_arms).T
    y = self.collected_rewards
    if self.iteration % self.refitFrequency == 0:
      self.gp.fit(x,y)
    self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
    self.means = np.array(self.means)
    self.sigmas = np.maximum(self.sigmas, 1e-2)

  def update(self, pulled_arm, reward):
    self.t += 1
    self.update_observations(pulled_arm, reward)
    self.update_model()

  def pull_arm(self):
    upper_conf = self.means + 1.96 * self.sigmas
    return np.random.choice(np.where(upper_conf == upper_conf.max())[0])


 
