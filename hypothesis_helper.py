# 
# Helper functions for hypothesis space
# 
import itertools
import time
from typing import Any, Callable, List, Tuple

import numpy as np


def get_correct_hypothesis(hypothesis_space: List[Tuple[Any]], 
                           hypothesis_fn: Callable, 
                           xs: np.array, ys: np.array):
    """
    Compute the set of hypothesis that is fully correct under the current set of 
    observations (xs, ys). This can be used to compute a posterior probability.

    Args:
        hypothesis_space: A list of hypotheses, length [num_hypothesis], 
                          each hypothesis is represented as a tuple of args for hypothesis_fn
        hypothesis_fn: A function that takes an observation and a hypothesis, 
                       and returns a prediction. Maps d -> {0, 1}.
        x (np.array): An array of input observations with shape (n, d)
        y (np.array): An array of output labels with shape (num_observations,).
    """
    assert len(xs) == len(ys)    

    ## Compute p(y=1 | x) for each hypothesis
    corrects = []
    for x, y in zip(xs, ys):
        pred = np.array([hypothesis_fn(x, *h_tup) for h_tup in hypothesis_space])  # [num_hypotheses,]
        pred_correct = (pred == y)  # [num_hypotheses,]
        corrects.append(pred_correct)
    corrects = np.stack(corrects, axis=0)  # [num_observations, num_hypotheses]
    all_correct = np.all(corrects, axis=0)  # [num_hypotheses,]

    # only consider this subset of all correct hypothesis
    #posterior_prob = all_correct / (all_correct.sum() + 1e-8)  # [num_hypotheses,]
    #return posterior_prob

    return all_correct


def hypothesis_func(x: np.array, blicket_id_mask: np.array, rule: str):
    """
    Computes f(x), where f is defined by blicket_id_mask and the rule
    """
    blicket_id_mask = np.array(blicket_id_mask, dtype=np.bool_)
    if rule == "conjunctive":
        return np.all(x[blicket_id_mask])
    elif rule == "disjunctive":
        return np.any(x[blicket_id_mask])
    else:
        raise ValueError(f"Unknown rule: {rule}")


def get_full_hypothesis_space(num_objects: int) -> List[Tuple]:
    """
    Generate the full hypothesis space for a given number of objects.
    """
    blicket_identities = list(itertools.product([0, 1], repeat=num_objects))
    blicket_identities = [identity for identity in blicket_identities if any(identity)]  # remove all zeros

    rules = ["conjunctive", "disjunctive"]
    hyp_space = list(itertools.product(blicket_identities, rules))
    return hyp_space


def compute_num_valid_hypothesis(obs: np.array):
    # assume only last element is used as the prediction target
    xs = obs[:, :-1]
    ys = obs[:, -1]

    hyp_space = get_full_hypothesis_space(xs.shape[1])
    all_correct = get_correct_hypothesis(hyp_space, hypothesis_func, xs, ys)
    return np.sum(all_correct)


if __name__ == "__main__":
    # Testing
    num_objects = 8
    num_samples = 16

    rng = np.random.default_rng(3)

    obs = rng.integers(0, 2, (num_samples, num_objects + 1))
    blick_mask = rng.integers(0, 2, (num_objects,))
    for i in range(num_samples):
        obs[i, -1] = hypothesis_func(obs[i, :-1], blick_mask, "conjunctive")

    start_time = time.time()    
    n_valid = compute_num_valid_hypothesis(obs)
    end_time = time.time()
    print(f"Number of valid hypothesis: {n_valid}")
    print(f"Time taken: {end_time - start_time} seconds")
    
