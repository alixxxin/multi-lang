import numpy as np
import pylab as pl
import ot
import scipy as sp
import scipy.stats
from pulp import *
from fractions import Fraction

import time

metric_choices = ['euclidean', 'sqeuclidean', 'cosine']

def build_MXY(X, Y, metric='euclidean'):
    """
    Build the cost matrix
    """
    return sp.spatial.distance.cdist(X, Y, metric)

def compute_wasserstein_distance(X, Y, a, bi, reg=1e-1):
    dist = 0
    for i in range(len(Y)):
        M = build_MXY(X, Y[i])
        plan = ot.bregman.sinkhorn(a, bi[i], M, reg)
        dist += np.sum(M*plan)
    return dist


def dual(M, a, b):
    max_i_in_m = np.amax(M, axis=1)
    max_j_in_m = np.amax(M, axis=0)
    prob = LpProblem("Example of standard maximum problem",LpMaximize)
    alpha = LpVariable.dicts("alpha", list(range(len(a))))
    beta = LpVariable.dicts("beta", list(range(len(b))))

    var_sum = [alpha[i]*a[i] for i in range(len(a))]
    var_sum.extend([beta[j]*b[j] for j in range(len(b))])
    obj = pulp.lpSum(var_sum)
    
    prob += obj
    for i in range(len(a)):
        for j in range(len(b)):
            prob += (alpha[i] + beta[j]) <= M[i][j], "constraint on "+str(i)+" "+str(j)
    prob += pulp.lpSum(alpha[i] for i in range(len(a))) == 0
    
    prob.solve()
    return prob

def opt_alpha_in_dual(M, a, b):
    prob = dual(M, a, b)
    dalpha = np.empty(len(a))
    for var in prob.variables():
        var_name, var_idx = var.name.split("_")
        if var_name == "alpha":
            var_idx = int(var_idx)
            dalpha[var_idx] = var.value()

    return dalpha

def barycenter_algorithm(X_orig, Yi_orig, bi, lambdas, tol=1e-8, metric='euclidean', reg=1e-2, maxiter=100):
    current_milli_time = lambda: int(round(time.time() * 1000))

    timec = current_milli_time()
    X = ot.lp.free_support_barycenter(Yi_orig, bi, X_orig)
    timea = current_milli_time()
    print("total time computed:"+ str(((timec-timea)/(1000*60))%60 ))
    a = compute_barycenter_weight(X, Yi_orig, bi, tol=tol)
    timeb = current_milli_time()
    print("total time computed:"+ str(((timeb-timea)/(1000*60))%60) )
    return X, a



def compute_barycenter_weight2(X, Y, bi, tol=1e-5, maxiter=10):
    """
    Nl: number of supports
    X: barycenter support location (Nl * d)
    Y: list of distributions support location (mi * d)
    b: list of weight on distributions (mi,)
    """
    N = len(Y)
    Nl = len(X)

    assert(len(Y) == len(bi))
    
    Mi = [build_MXY(X, Y[i]) for i in range(0, N)]
    
    a1 = ot.unif(Nl)
    a2 = ot.unif(Nl)
    a = ot.unif(Nl)
    differ = 1
    t = 1
    d = compute_wasserstein_distance(X, Y, a1, bi)
    
    while differ > tol and t < maxiter:
        preva = a
        beta = (t + 1) / 2

        subgradient = 0
        for i in range(N):
            subgradient += opt_alpha_in_dual(Mi[i], a, bi[i])
        subgradient = subgradient / N
        a2 = a2 * (np.exp(-(beta) * subgradient))
        a2 = a2 / np.sum(a2)
        a1 = (1 - (1 / beta)) * a1 + (1 / beta) * a2
        a = (1 - (1 / beta)) * a1 + (1 / beta) * a2
        differ = np.sum(np.square(a - preva))
        t = t+1
    return a

def unweighted_barycenter(X_orig, Yi_orig, bi, tol=1e-8, metric='euclidean', reg=1e-2, maxiter=100):
    X, a = ot.lp.free_support_barycenter(Yi_orig, bi, X_orig), ot.unif(len(X_orig))
    return X, a


def cal_barycenter_for(ls, test_func, Cs, ps, lambdas, X_init, tol=1e-8, metric='euclidean', reg=1e-2, maxiter=100, bregmanmaxiter=30):
    print(X_init.shape)
    X, a = weighted_barycenter_algorithm(ls, test_func, X_init, Cs, ps, lambdas, tol=tol, metric=metric, reg=reg, maxiter=maxiter, bregmanmaxiter=bregmanmaxiter)
    return X, a


def weighted_barycenter_algorithm(ls, test_func, X_orig, Yi_orig, bi, lambdas, tol=1e-8, metric='euclidean', reg=1e-2, maxiter=20, bregmanmaxiter=30):
    """
    k : number of supports in X
    X_orig  : init of barycenter (k * d)
    Yi_orig : list of distributions size (k_i * d)
    bi : list of weights size (k_i)
    tol: tolerance
    """
    assert(len(Yi_orig) == len(bi))
    assert(len(X_orig[0]) == len(Yi_orig[0][0]))
    
    X = X_orig
    Yi = Yi_orig
    
    displacement = 1
    niter = 0

    while (displacement > tol and niter < maxiter):
        X_prev = X
        a = compute_barycenter_weight(X, Yi, bi, lambdas, tol=tol, maxiter=bregmanmaxiter, reg=reg)
        Tsum = np.zeros(X.shape)

        for i in range(0, len(bi)): 
            M = build_MXY(X, Yi[i], metric=metric)
            #T = ot.sinkhorn(a, bi[i], M, reg)
            T = ot.emd(a, bi[i], M)
            Tsum = Tsum + lambdas[i] * np.reshape(1. / a, (-1, 1)) * np.matmul(T, Yi[i])

        displacement = np.sum(np.square(Tsum - X))

        print("~~~~epoch "+str(niter)+"~~~~")

        #i = ls.index('en')
        #for j in range(len(ls)):
        #    if i!=j and (not ls[i].isdigit()) and (not ls[j].isdigit()):
        #        mapping = ot.emd(bi[i], a, build_MXY(Yi[i], X))
        #        mapping2 = ot.emd(a, bi[j], build_MXY(X, Yi[j]))
        #        print("="*20+"begin testing mapping for "+ls[i]+" and "+ls[j]+"="*21)
        #        test_func(ls[i], ls[j], np.dot(mapping, mapping2))
        #        mapping = None
        #        mapping2 = None

        #for i in range(len(ls)):
        #    for j in range(len(ls)):
        #        if i!=j and (not ls[i].isdigit()) and (not ls[j].isdigit()):
        #            mapping = ot.emd(bi[i], a, build_MXY(Yi[i], X)) 
        #            mapping2 = ot.emd(a, bi[j], build_MXY(X, Yi[j])) 
        #            print("="*20+"begin testing mapping for "+ls[i]+" and "+ls[j]+"="*21)
        #            try:
        #                test_func(ls[i], ls[j], np.dot(mapping, mapping2))
        #            except:
        #                print("failed to eval on "+ls[i]+" and "+ls[j])
        #            mapping = None
        #            mapping2 = None


        X = Tsum
        niter += 1

    return X, a


def compute_barycenter_weight(X, Y, bi, lambdas, tol=1e-5, maxiter=25, reg=1e-1):
    assert(len(Y) == len(bi))
    
    Cs = [build_MXY(X, Y[i]) for i in range(len(Y))]
    K = [np.exp(-Cs[i] / reg) for i in range(len(Cs))]

    cpt = 0
    err = 1

    uKv = np.array([np.dot(K[i], np.divide(bi[i], np.sum(K[i], axis=0))) for i in range(len(K))])
    u = (geometricMean(uKv) / uKv.T).T
    
    while (err > tol) and (cpt < maxiter):
        cpt += 1
        uKv = np.array([u[i] * (np.dot(K[i], np.divide(bi[i], np.dot(K[i].T, u[i])))) for i in range(len(K))]) 
        gbar = geometricBar(lambdas, uKv.T)
        u = np.array( [ ( u[i] * gbar ) / uKv[i] for i in range(len(uKv))])
        err = np.sum(np.std(uKv, axis=1))

    return geometricBar(lambdas, uKv.T)


def geometricBar(weights, alldistribT):
    """return the weighted geometric mean of distributions"""
    assert(len(weights) == alldistribT.shape[1])
    return np.exp(np.dot(np.log(alldistribT), weights.T))


def geometricMean(alldistribT):
    """return the  geometric mean of distributions"""
    return np.exp(np.mean(np.log(alldistribT), axis=1))





