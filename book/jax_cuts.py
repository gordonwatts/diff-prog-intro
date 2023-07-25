import jax.numpy as jnp
import jax
from typing import Callable


def cut_erf(cut: float, data):
    "Take a jax array and calculate an error function on it"
    return (jax.lax.erf(data - cut) + 1) / 2.0


def loss_sig_sqrt_b(f, sig_j, back_j):
    "Calculate the S/sqrt(B) for two 1D numpy arrays with the cut at cut."

    # Weight the data and then do the sum
    wts_sig = f(sig_j)
    wts_back = f(back_j)

    S = jnp.sum(wts_sig)
    B = jnp.sum(wts_back)

    return S / jnp.sqrt(B)


def loss_x_entropy(f, sig_j, back_j):
    '''Binary x-entropy for this function

    Args:
        f (Callable): The function that will calculate the prop of sig/back
        sig_j (array): Signal values
        back_j (array): Background values
    '''
    entropy_sig = -jnp.log(f(sig_j) + 1e-6)
    entropy_back = -jnp.log(1-f(back_j) + 1e-6)
    return jnp.sum(entropy_sig) + jnp.sum(entropy_back)


def loss_squares(f: Callable, sig_j, back_j):
    '''Calculate the loss which is the sum of the squares of the difference.

    Args:
        f (Callable): Cut we are to measure the loss against
        sig_j (array): Signal data
        back_j (array): Background Data
    '''
    return jnp.sum(jnp.square((1-f(sig_j)))) + jnp.sum(jnp.square(f(back_j)))


def cut_sigmoid(cut: float, data):
    slope = 1.0
    return 1 / (1 + jnp.exp(-slope * (data - cut)))


def cut_sigmoid_balanced(cut: float, data):
    slope = 1.0
    s = 1 / (1 + jnp.exp(-slope * (data - cut)))
    return s*2.0 - 1.0


def cut_hard(cut: float, data):
    return jnp.sign(jnp.sign(data-cut) + 1.0)
