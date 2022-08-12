import numpy as np

class Perceptron:
    def __init__(self, eta=0.1, n_iter=50, rand_state=1): 
        ''' 
           eta - learning rate, 
           n_iter - number of iterations,
           rand_state - param for np.random.RandomState
        '''
        self.eta = eta
        self.n_iter = n_iter
        self.rand_state = rand_state
    def fit(self, X, y):
        rgen = np.random.RandomState(self.rand_state) # generator
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # generation of small random values
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = eta*(target - self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
        return self
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def predict(self, X):
        return np.where(net_input(X) >= 0.0, 1, -1)
