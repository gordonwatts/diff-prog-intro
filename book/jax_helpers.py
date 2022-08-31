import jax
import jax.numpy as jnp
import numpy as numpy

from samples import data_sig, data_back, sig_avg, sig_width

# Convert the data to jax arrays
data_sig_j = jnp.asarray(data_sig)
data_back_j = jnp.asarray(data_back)

def erf(data, cut: float):
    'Take a jax array and calculate an error function on it'
    return (jax.lax.erf(data-cut)+1)/2.0

def sig_sqrt_b(cut):
    'Calculate the S/sqrt(B) for two 1D numpy arrays with the cut at cut.'

    # Weight the data and then do the sum
    wts_sig = erf(data_sig_j, cut)
    wts_back = erf(data_back_j, cut)

    S = jnp.sum(wts_sig)
    B = jnp.sum(wts_back)

    return S/jnp.sqrt(B)

data_sig_big = numpy.random.normal(sig_avg, sig_width, len(data_back))

training_data = jnp.concatenate((data_back, data_sig))
training_truth = jnp.concatenate((jnp.zeros(len(data_back)), jnp.ones(len(data_sig))))

def predict(data, c: float):
    'Predict if background or signal depending on the cut'
    return erf(data, c)

def loss(c: float, data = training_data, truth = training_truth):
    'Calculate the standard distance loss'
    return jnp.sum((predict(data, c) - truth)**2)
