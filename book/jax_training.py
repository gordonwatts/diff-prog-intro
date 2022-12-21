import haiku as hk
from typing import Optional, List, Callable, Dict
import jax
import jax.numpy as jnp
import optax
from optax import Params
import math


class Selection(hk.Module):
    """Apply a selection cut to each input, output is a weight,
    zero if the cut would reject, one if it would not"""

    def __init__(
        self,
        f_cut: Callable,
        initial_cuts: Optional[List[float]] = None,
        name="SelectionCut",
    ):
        super().__init__(name=name)
        self._initial_cuts = initial_cuts
        self._f_cut = f_cut

    def __call__(self, x):
        "Apply a selection cut for all the incoming data."

        # See if we have a decent set of initializers
        cuts_initial = (
            jnp.asarray(self._initial_cuts)
            if self._initial_cuts is not None
            else jnp.ones(x.shape[1])
        )
        assert (
            cuts_initial.shape[0] == x.shape[1]
        ), f"Incorrect number of initial cut values specified - need {x.shape[1]}"

        # Get the parameters to apply here
        cuts = hk.get_parameter(
            "cuts",
            shape=[x.shape[1]],
            dtype=jnp.float32,
            init=lambda shp, dtyp: cuts_initial,
        )

        # Next, apply the cut
        cut_data_pairing = [
            (cuts[index], x[:, index]) for index in range(0, x.shape[1])
        ]
        wts = jnp.stack(
            [self._f_cut(c, x_col) for c, x_col in cut_data_pairing], axis=1
        )

        return jnp.prod(wts, axis=1)


def default_loss(preds, actual):
    upped = jax.nn.sigmoid(preds)
    print(preds.shape, actual.shape)
    return optax.softmax_cross_entropy(upped, actual)


def train(
    model,
    key,
    epochs,
    training_data,
    training_truth,
    learning_rate=0.002,
    use_loss_function=default_loss,
) -> Dict:
    """Run the training with the specified loss and model.

    - Simple loop with gradient feedback, one iteration per epoch
    - uses the `optax.adam` optimizer with the given learning rate.

    Args:
        model (_type_): The model to train
        key (_type_): Random number key for initalization
        epochs (_type_): How many training epochs to run
        training_data (_type_): The training data vector (signal and background)
        training_truth (_type_): A 1D vector that indicates signal or background for
                                 each training data entry.
        learning_rate (float, optional): How quickly to adjust the adam optimizer.
                                         Defaults to 0.002.

    Returns:
        Dict: The final training parameters.
    """
    # Initialize the weights
    key, _ = jax.random.split(key)
    params = model.init(key, training_data)  # type: optax.Params

    # Init the optimizer
    opt_init, opt_update = optax.chain(optax.adam(learning_rate), optax.zero_nans())
    opt_state = opt_init(params)

    # Build the loss function
    def loss_func(weights, input_data, actual):
        "Calc the loss function"
        preds = model.apply(weights, key, input_data)
        preds = preds.squeeze()
        return use_loss_function(preds, actual)
        # preds = jax.nn.sigmoid(preds)
        # return optax.softmax_cross_entropy(preds, actual)

    neg_loss_func = jax.jit(jax.value_and_grad(loss_func))

    # Train
    report_interval = int(epochs / 10)
    old_params: Optional[Params] = None
    old_loss = 1000
    bad_loss = False
    for i in range(1, epochs + 1):
        loss, param_grads = neg_loss_func(params, training_data, training_truth)
        updates, opt_state = opt_update(param_grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if math.isnan(loss):
            print("WARNING: Loss is nan - returning last good epoch")
            assert (
                old_params is not None
            ), "Fatal error - did not make it a single iteration"
            params = old_params
            loss = old_loss
            bad_loss = True

        if i % report_interval == 0 or i == 1 or bad_loss:
            print(f"NegLogLoss : {loss:.2f}, epoch: {i}")

        if bad_loss:
            break

        old_params = params
        old_loss = loss

    assert params is not None
    return params  # type:ignore


def train_cut(
    t_data,
    t_truth,
    f_cut: Callable,
    epochs=10,
    initial_cuts=(1.0, 1.0),
    use_loss_function=default_loss,
):
    """Simplify training a cut by setting default parameters for everything else.

    Args:
        f_cut (_type_): The Callable that can perform the cut
    """

    def model_builder(x):
        cuts = Selection(f_cut, initial_cuts=initial_cuts)
        return cuts(x)

    model = hk.transform(model_builder)
    key = jax.random.PRNGKey(1234)

    return train(
        model, key, epochs, t_data, t_truth, use_loss_function=use_loss_function
    )
