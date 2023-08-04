import numpy as np
import matplotlib.pyplot as plt
import logging

vals = np.loadtxt("values.txt")

#cut the first third of points to reduce counting extractions made when the Markov chain is not in equilibrium
idx_cut = len(vals)//3
vals_cut = vals[idx_cut:]

#evaluate mean and "normal" variance of the mean of the full set
mean = np.mean(vals)
var = np.var(vals)/len(vals)

#evaluate mean and "normal" variance of the cut set
mean_cut = np.mean(vals_cut)
var_cut = np.var(vals_cut)/len(vals_cut)

#print out results
print(f"mu (full)= {mean} +- {np.sqrt(var)}")
print(f"mu (cut) = {mean_cut} +- {np.sqrt(var_cut)}")

#plot out the full Markov process, showing the cut line
plt.figure()
plt.plot(vals)
plt.vlines(idx_cut,0,max(vals))
plt.show()
