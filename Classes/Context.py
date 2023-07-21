import numpy as np


class Context():
  def __init__(self, features, ts_learner, gpts_learner, gpucb_learner):
    self.features = features
    self.ts_learner = ts_learner
    self.gpts_learner = gpts_learner
    self.gpucb_learner = gpucb_learner
