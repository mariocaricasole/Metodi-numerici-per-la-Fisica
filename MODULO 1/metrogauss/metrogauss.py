import numpy as np

#define parameters and starting value
xk = 0
L=10000
step = 1
delta = 0.1
mu = 5
sigma = 1
vals = []
means = []

#define Gaussian probability function
def p(x,mu,sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2))

#run through Markov chain implementing Metropolis
for i in range(L):
    #define symmetric interval
    left,right = xk - delta, xk + delta

    #choose random point on symmetric interval, first extracting in [0,1] and then transforming to match the interval
    xp = np.random.random()*2*delta + left

    #accept-reject step
    r = min(1,p(xp,mu,sigma)/p(xk,mu,sigma))
    if(np.random.random() < r):
        xk = xp

    vals.append(xk)
    # means.append(np.mean(vals))

np.savetxt("values.txt",vals)
# np.savetxt("means.txt",means)
