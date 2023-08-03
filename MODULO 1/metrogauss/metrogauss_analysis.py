import numpy as np
import matplotlib.pyplot as plt
import logging

vals = np.loadtxt("values.txt")

#evaluate mean and "normal" standard deviation
mean = np.mean(vals)
var = np.var(vals)

print(f"Sample mean: {mean}")
print(f"Sample variance: {var}")
