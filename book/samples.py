import numpy as np


# Fixing random state for reproducibility
np.random.seed(19680801)

# Sizes of the samples
N_sig = 10000
N_back = 10000

sig_avg = 5.0
sig_width = 1.0

# The samples for use
data_sig = np.random.normal(sig_avg, sig_width, size=N_sig)
data_back = np.random.normal(0.0, 4.0, size=N_back)
