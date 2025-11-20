import pandas as pd
import numpy as np

def init_adam_state(intercepts, coefs):
    state = []

    for intercept, coef in zip(intercepts, coefs):
        layer_state = {
            "m_b": np.zeros_like(intercept),
            "v_b": np.zeros_like(intercept),
            "m_w": np.zeros_like(coef),
            "v_w": np.zeros_like(coef),
        }
        state.append(layer_state)

    return {
        "t": 0,
        "layers": state
    }


def adam_update(
    X: np.ndarray,
    intercept: np.ndarray,
    coef: np.ndarray,
    error: np.ndarray,
    da_dz: np.ndarray | None,
    layer_state: dict,
    adam_state: dict,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):
    """
    Perform a single Adam update step.
    """
    if da_dz is None:
        delta = error
    else:
        delta = error * da_dz

    d_intercept = np.mean(delta, axis=0)
    d_coef = (delta.T @ X) / X.shape[0]

    adam_state["t"] += 1
    t = adam_state["t"]

    layer_state["m_b"] = beta1 * layer_state["m_b"] + (1 - beta1) * d_intercept
    layer_state["v_b"] = beta2 * layer_state["v_b"] + (1 - beta2) * (d_intercept ** 2)

    m_b_hat = layer_state["m_b"] / (1 - beta1 ** t)
    v_b_hat = layer_state["v_b"] / (1 - beta2 ** t)

    intercept -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

    layer_state["m_w"] = beta1 * layer_state["m_w"] + (1 - beta1) * d_coef
    layer_state["v_w"] = beta2 * layer_state["v_w"] + (1 - beta2) * (d_coef ** 2)

    m_w_hat = layer_state["m_w"] / (1 - beta1 ** t)
    v_w_hat = layer_state["v_w"] / (1 - beta2 ** t)

    coef -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

    return intercept, coef


def back_propagation_bonus(batch_x, batch_y, z_list, a_list,
                     da_dz_list, intercepts, coefs, config, adam_state):
    """
    Backward pass using Adam optimizer.
    """
    error = a_list[-1] - batch_y
    error_prop = error

    for layer in reversed(range(len(config["layer"]))):

        if layer == 0:
            input_x = batch_x
        else:
            input_x = a_list[layer - 1]

        layer_state = adam_state["layers"][layer]

        intercept, coef = adam_update(
            input_x,
            intercepts[layer],
            coefs[layer],
            error_prop,
            da_dz_list[layer],
            layer_state,
            adam_state,
            lr=config["learning_rate"],
            beta1=config.get("beta1", 0.9),
            beta2=config.get("beta2", 0.999),
            eps=config.get("epsilon", 1e-8)
        )

        if layer > 0:
            error_prop = (
                error_prop @ np.array(coefs[layer])
            ) * da_dz_list[layer - 1]

        intercepts[layer] = intercept
        coefs[layer] = coef

    return intercepts, coefs
