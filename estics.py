import numpy as np

import math

n=157
t=92
r= 30
lamuda = 1e-3
MaxIter = 50


X = np.loadtxt("out.txt")
B = np.loadtxt("onezero-Matrix.txt")
S = X*B


def myInverse(B,L,lamuda,S):
	n = B.shape[0]
	t = B.shape[1]
	r = L.shape[1]
	Y = np.zeros((r,t))
	
	for i in range(0,t):
		Bi = np.diag(B[:,i])
		P = np.concatenate((np.dot(Bi,L),np.power(lamuda,1/2)*np.eye(r)),axis=0)
		Zr = np.zeros((r,1))
		Si = S[:,i].reshape((-1,1))
		Q = np.concatenate((Si,Zr),axis=0)
		Y[:,i] = np.linalg.solve(np.dot(P.T,P),np.dot(P.T,Q)) .reshape((r))
		
	return Y

		

def ESTI_CS(S,B,r,lamuda,MaxIter):
	v=1e50
	Lopt = np.random.rand(n,r)
	# Lopt = np.ones((n,r))	
	for i in range(1,MaxIter):
		Ropt=myInverse(B,Lopt,lamuda,S).T
		Lopt=myInverse(B.T,Ropt,lamuda,S.T).T
		X=np.dot(Lopt,Ropt.T)
		
		vopt1=np.linalg.norm(np.multiply(B,X)-S)
		vopt2=lamuda*(np.linalg.norm(Lopt)+np.linalg.norm(Ropt.T))
		vopt=vopt1+vopt2	
		if vopt<v:
			v=vopt
			L=Lopt
			R=Ropt
		
	X_min = np.dot(L,R.T)
	return X_min

X_result = ESTI_CS(S,B,r,lamuda,MaxIter)
np.savetxt("result.txt",X_result)



