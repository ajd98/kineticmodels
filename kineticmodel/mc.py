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
        t = 1/(transmat[transmat > 0].min())*1.0e+4
        try:
            result = self.time_vec.dot(scipy.linalg.expm(transmat*t))
        except ValueError:
            print(transmat)
            print(transmat*t)
            raise Exception 

        # find left eigenvectors
        # first, make stochastic matrix
        #stochmat = numpy.copy(transmat)
        #for i in xrange(stochmat.shape[0]):
        #    # row normalize
        #    stochmat[i] /= stochmat[i].sum()
        #print(stochmat)
        #eigvals, eigvecs = scipy.linalg.eig(stochmat, left=True, right=False)
        #where_one = numpy.argmax(eigvals)
        #print(eigvals[where_one])

        #result = eigvecs[where_one]

        return result/result.sum()

    def transmat_to_generator(self, transmat):
        for i in xrange(transmat.shape[0]):
            transmat[i,i] = -1*(transmat[i].sum()-transmat[i,i])
        return transmat

    def calculate_stationary_distribution(self, alpha=0.05, nestimates=1000):
        '''
        Calculate the stationary distribution of the markov chain, along with
        uncertainties. Return m, l, u, vectors representing the maximum
        likelihood estimate, as well as the upper and lower ends of a 95%
        credible interval.
        '''
        estimates = numpy.empty((nestimates,self.time_vec.shape[0]))
        for i in xrange(nestimates):
            transmat = self.estimator.random_estimate()
            generatormat = self.transmat_to_generator(transmat)
            estimates[i] = self._solve(generatormat)
        l = numpy.empty(self.time_vec.shape)
        u = numpy.empty(self.time_vec.shape)
        for i in xrange(self.time_vec.shape[0]):
            estimates[:,i].sort()
            l[i] = estimates[:,i][int(alpha/2.*nestimates)]
            u[i] = estimates[:,i][int((1-alpha/2.)*nestimates)]
        transmat = numpy.require(numpy.copy(self.count_matrix), dtype=float)
        for i in xrange(self.time_vec.shape[0]):
            if self.time_vec[i] > 0:
                transmat[i,:] /= self.time_vec[i]
        generatormat = self.transmat_to_generator(transmat)
        m = self._solve(generatormat)
        return m, l, u



