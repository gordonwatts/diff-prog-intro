import jax
import jax.numpy as jnp

from samples import data_sig, data_back

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
