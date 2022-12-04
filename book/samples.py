import numpy as np


# Fixing random state for reproducibility
np.random.seed(19680801)

# Sizes of the samples. Keep same size to help
# training - as reweighing will happen anyway.
N_sig = 10000
N_back = 10000

sig_avg = 5.0
sig_width = 1.0

# The 1D samples for use
data_sig = np.random.normal(sig_avg, sig_width, size=N_sig)
data_back = np.random.normal(0.0, 4.0, size=N_back)

# The 2D samples for use
data_2D_sig = np.random.multivariate_normal(
    [1.5, 2.0], [[0.5, 0.2], [0.2, 0.5]], [N_sig]
)
data_2D_back = np.random.multivariate_normal(
    [0.0, 0.0], [[9.0, 0.0], [0.0, 9.0]], [N_back]
)
