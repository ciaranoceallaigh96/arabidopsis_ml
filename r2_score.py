#takes plink .profile file as first input and cv number as second 
#should be the same result as a linear regression between score and prediction 
import sys
import numpy as np
print(sys.argv[1])
data = np.loadtxt(sys.argv[1], skiprows=1, usecols=[2,5])
r2_score = str(np.corrcoef(data[:,0], data[:,1])**2)
print("Result for cv %s is R2 %s" % (sys.argv[2], r2_score))
