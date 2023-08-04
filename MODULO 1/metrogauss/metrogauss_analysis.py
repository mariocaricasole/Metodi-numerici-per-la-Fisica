import numpy as np
import matplotlib.pyplot as plt
import logging

##### PRINT FUNCTION ####
def printRes(specifier,mean,var):
    print(f"mu ({specifier}) = {mean} +- {np.sqrt(var)}")

#load extractions from file
vals = np.loadtxt("values.txt")

##### NAIVE ANALYSIS #####
#evaluate mean and "normal" variance of the mean of the full set
mean = np.mean(vals)
var = np.var(vals,ddof=1)/len(vals)

##### DISCARDING INITIAL DATA POINTS #####
#cut the first third of points to reduce counting extractions made when the Markov chain is not in equilibrium
idxCut = len(vals)//3
valsCut = vals[idxCut:]

#evaluate mean and "normal" variance of the cut set
meanCut= np.mean(valsCut)
varCut = np.var(valsCut,ddof=1)/len(valsCut)


##### WITH AUTOCORRELATION ESTIMATE #####
#define autocorrelation function
def C(k,data):
    total = 0.
    N = len(data)
    dataMean = mean
    for i in range(0,N-k):
        total += (data[i] - dataMean)*(data[i+k] - dataMean)

    return total/(N-k)

#define autocorrelation time
def tau(data):
    total = 0.
    N = len(data)
    for k in range(1,N):
        total+=C(k,data)

    return total

#evaluate mean and variance, adjusted for autocorrelation
meanAC = meanCut
tau = tau(valsCut)
varAC = varCut*(1+2*tau)

#print out results
printRes("full", mean, var)
printRes("cut", meanCut, varCut)
printRes("autocorr", meanAC, varAC)

#plot out the full Markov process, showing the cut line
plt.figure()
plt.plot(vals)
plt.vlines(idxCut,0,max(vals))
plt.show()
