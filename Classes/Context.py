class Context():
    def __init__(self, features, ts_learner_gpts, n_gpts_learner, cc_gpts_learner, ts_learner_gpucb, n_gpucb_learner,
                 cc_gpucb_learner):
        self.features = features
        self.ts_learner_gpts = ts_learner_gpts
        self.n_gpts_learner = n_gpts_learner
        self.cc_gpts_learner = cc_gpts_learner
        self.ts_learner_gpucb = ts_learner_gpucb
        self.n_gpucb_learner = n_gpucb_learner
        self.cc_gpucb_learner = cc_gpucb_learner
