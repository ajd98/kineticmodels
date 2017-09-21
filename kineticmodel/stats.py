#!/usr/bin/env python
import numpy
import scipy.special

class BayesianEstimatorPoisson(object):
    def __init__(self, time_vec, count_matrix):
        self.count_matrix = count_matrix
        self.time_vec = time_vec

    def likelihood(self, k):
        # 3 -> 1
        t = self.time_vec
        n = self.count_matrix
        
        val = (k*t)**n*numpy.exp(-k*t)/scipy.special.gamma(n+1)
        # normalize
        val = val*t_tot
        return val

    def random_estimate(self):
        '''
        Return an n-by-n matrix wherer the i,j element represents the transition
        rate from state i to state j.  Each element is chosen randomly from the
        likelihood (the posterior probability for the transition rate, given
        that self.count_matrix events were observed in period time_vec, using
        uniform prior).
        '''
        estimate = numpy.empty(self.count_matrix.shape, dtype=float)
        for i in xrange(self.count_matrix.shape[0]):
            for j in xrange(self.count_matrix.shape[1]):
                n = self.count_matrix[i,j]
                t = self.time_vec[i]
                if n >= 0 and t > 0:
                    estimate[i,j] = numpy.random.gamma(n+1,scale=1./t) 
                else:
                    estimate[i,j] = 0
        return estimate
