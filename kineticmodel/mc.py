#!/usr/bin/env python
import numpy
import scipy.linalg
import stats

class CtsTimeMC(object):
    def __init__(self, time_vec, count_matrix):
        '''
        time_vec: length-n vector of times spent in each of n states
        count_matrix: n-by-n matrix of transition counts, where the i,j element
            is the number of transitions from state i to state j
        '''
        self.time_vec = time_vec
        self.count_matrix = count_matrix
        self.estimator = stats.BayesianEstimatorPoisson(time_vec, count_matrix)

    def _solve(self, transmat):
        '''
        transmat: n-by-n matrix of transition rates, where the i,j element is
            the transition rate from state i to state j
        '''
        # We know P(t) = exp(M*t), where M is the transition matrix.
        # find the time for very larget
        t = 1/transmat[transmat > 0].min()*1e+6
        return scipy.linalg.expm(transmat*t)

    def calculate_stationary_distribution(self, alpha=0.05, nestimates=1000):
        '''
        Calculate the stationary distribution of the markov chain, along with
        uncertainties. Return m, l, u, vectors representing the maximum
        likelihood estimate, as well as the upper and lower ends of a 95%
        credible interval.
        '''
        estimates = numpy.empty((nestimates,self.time_vec.shape[0]))
        for i in xrange(nestimates):
            estimates[i] = self._solve(self.estimator.random_estimate())
        l = numpy.empty(self.time_vec.shape)
        u = numpy.empty(self.time_vec.shape)
        for i in xrange(self.time_vec.shape[0]):
            l[i] = estimates[:,i].sort()[int(alpha/2.*nestimates)]
            u[i] = estimates[:,i].sort()[int((1-alpha/2.)*nestimates)]
        transmat = self.count_matrix
        for i in xrange(self.time_vec.shape[0]):
            transmat[i,:] /= self.timevec[i]
        m = self._solve(transmat)
        return m, l, u



