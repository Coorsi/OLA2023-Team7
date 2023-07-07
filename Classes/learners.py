import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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


class GPTS_Learner(Learner):
  def __init__(self, n_arms, arms):
    super().__init__(n_arms)
    self.arms = arms
    self.means = np.zeros(n_arms)
    self.sigmas = np.ones(n_arms)*10
    self.pulled_arms = []
    alpha = 1
    kernel = C(1e1, (1e-7, 1e7)) * RBF(1e1, (1e-7, 1e7))
    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2)
    self.iteration = 0

  def update_observations(self, pulled_arm, reward):
    super().update_observations(pulled_arm, reward)
    self.pulled_arms.append(self.arms[pulled_arm])
  
  def update_model(self):
    self.iteration += 1
    x = np.atleast_2d(self.pulled_arms).T
    y = self.collected_rewards
    if sum([int(k) for k in str(self.iteration)]) < 6:
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
<<<<<<< HEAD
    kernel = C(1e1, (1e-7, 1e7)) * RBF(1e1, (1e-7, 1e7))
    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2,n_restarts_optimizer=5)
=======
    kernel = C(1e1, (1e-3, 1e3)) * RBF(1e1, (1e-3, 1e3))
    self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha**2)
    #self.bheta = 2*np.log((self.n_arms*np.power(self.t,2)*np.power(np.pi,2))/(6*0.05)) #0.05 is delta
    self.iteration = 0
>>>>>>> c82e2cfe00d3092914c16cab104d6d76f0dc0412

  def update_observations(self, pulled_arm, reward):
    super().update_observations(pulled_arm, reward)
    self.pulled_arms.append(self.arms[pulled_arm])
  
  def update_model(self):
    self.iteration += 1
    x = np.atleast_2d(self.pulled_arms).T
    y = self.collected_rewards
    if sum([int(k) for k in str(self.iteration)]) < 6:
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
<<<<<<< HEAD

  
 
=======
>>>>>>> c82e2cfe00d3092914c16cab104d6d76f0dc0412
