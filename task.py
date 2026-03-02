import numpy as np
x = np.array([10,20,30,40,50])
print(x)
sd = np.std(x)
print(sd)
avg = np.mean(x)
print(avg)
standardized = (x-avg)/sd
print(standardized)
