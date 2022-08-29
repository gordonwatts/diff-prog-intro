import numpy as np


# Fixing random state for reproducibility
np.random.seed(19680801)

# Sizes of the samples
N_sig = 1000
N_back = 10000

# The samples for use
data_sig = np.random.normal(5.0, 1.0, size=N_sig)
data_back = np.random.normal(0.0, 4.0, size=N_back)
