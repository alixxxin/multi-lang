import numpy as np
import sys
import math
from numpy.linalg import matrix_power
import scipy.linalg

#zero-centered Gaussians

def random_sym_psd_matrix(n):
	A=np.matrix(np.random.random((n,n)))
	Q=scipy.linalg.orth(A)
	D=np.diag(np.random.random(n)**2)
	PSD=Q.dot(D).dot(Q.T)
	return PSD

#fractional powers. For now, just invert the eigvec matrix
def mat_power(M,p):
	Lambda,V=np.linalg.eigh(M)
	Lambdap=Lambda**p
	Vinv=np.linalg.inv(V)
	Mp=V.dot(np.diag(Lambdap)).dot(Vinv)
	return Mp

def get_transport_map(P,Q):

	Proot=mat_power(P,0.5)
	matprod=Proot.dot(Q).dot(Proot)
	matprodroot=mat_power(matprod,0.5)
	invProot=mat_power(P,-0.5)
	A=invProot.dot(matprodroot).dot(invProot)
	return A



def calc_wasserstein_dist_gaussian(P,Q):

	Proot=mat_power(P,0.5)
	matprod=Proot.dot(Q).dot(Proot)
	matprodroot=mat_power(matprod,0.5)
	matsum=P+Q-2.0*matprodroot
	tr=matsum.trace()
	assert tr>-0.0000000001
	if tr<0:
		tr=0
	return np.sqrt(tr)

def barycenter_iter(Plist,lambdas,Mu):
	Alist=[]
	MeanMap=np.zeros(Mu.shape)
	for i in range(len(Plist)):
		A=get_transport_map(Mu,Plist[i])
		Alist.append(A)
		MeanMap+=lambdas[i]*A
	NewMu=MeanMap.dot(Mu).dot(MeanMap.T)
	return NewMu

def calc_score(Plist,lambdas,Mu):
	scores=[lambdas[i]*calc_wasserstein_dist_gaussian(Plist[i],Mu)**2 for i in range(len(Plist))]
	return sum(scores)


def calc_barycenter(Plist,lambdas,tolerance=0.0001):
	Mu=Plist[0]
	score=calc_score(Plist,lambdas,Mu)
	print("score:"+str(score))
	prevscore=2.0*score
	while abs(score-prevscore)>tolerance*score:
		Mu=barycenter_iter(Plist,lambdas,Mu)
		prevscore=score
		score=calc_score(Plist,lambdas,Mu)
		print("score:"+str(score))
	return Mu



#P=np.matrix([[1.0,0.0],[0.0,1.0]])
#Q=np.matrix([[1.1,0.0],[0.0,1.0]])

ndim=5

P=random_sym_psd_matrix(ndim)
Q=random_sym_psd_matrix(ndim)
R=random_sym_psd_matrix(ndim)
A=get_transport_map(P,Q)
print(np.sum(A.dot(P).dot(A.T)-Q))
A2=get_transport_map(Q,P)
print(np.sum(A.dot(A2.T)-np.eye(ndim)))

PQmean=(P+Q)*0.5

Mu=calc_barycenter([P,Q],[0.5,0.5])
