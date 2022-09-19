from typing import Dict
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

# Series of helpers for ML
def train(model, key, epochs, training_data, training_truth) -> Dict:
    # Init
    learning_rate = jnp.array(0.002)

    # Initialize the weights
    key, _ = jax.random.split(key)
    params = model.init(key, training_data)

    def NegLogLoss(weights, input_data, actual):
        'Calc the loss function'
        preds = model.apply(weights, key, input_data)
        preds = preds.squeeze()
        preds = jax.nn.sigmoid(preds)
        return (- actual * jnp.log(preds) - (1 - actual) * jnp.log(1 - preds)).mean()

    def UpdateWeights(weights,gradients):
        return weights - learning_rate * gradients

    # Build the loss function
    neg_loss_func = jax.jit(jax.value_and_grad(NegLogLoss))

    # Train
    report_interval = int(epochs / 10)
    for i in range(1,epochs+1):
        loss, param_grads = neg_loss_func(params, training_data, training_truth)
        params = jax.tree_map(UpdateWeights, params, param_grads)

        if i%report_interval == 0 or i == 1:
            print(f"NegLogLoss : {loss:.2f}, epoch: {i}")

    return params