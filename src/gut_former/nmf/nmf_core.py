import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization

from gut_former.utils.metrics import cos_sim, exp_var
from joblib import Parallel, delayed

log = logging.getLogger(__name__)


def nmf(X, n_components, init="nndsvdar", update_H=True, H=None, random_state=None):
    """Wrapper around non_negative_factorization with fixed settings.

    Returns (W, H, n_iter). Fixed params: solver=mu, beta_loss=kullback-leibler, max_iter=2000.
    """
    W, H, n_iter = non_negative_factorization(
        X,
        n_components=n_components,
        init=init,
        solver="mu",
        beta_loss="kullback-leibler",
        max_iter=2000,
        update_H=update_H,
        H=H,
        random_state=random_state,
    )
    return W, H, n_iter

def nmf_job(X_train, X_val, n_signatures, seed):
    W_train, H, _ = nmf(X_train, n_signatures, random_state=seed)
    W_val,   _, _ = nmf(X_val,   n_signatures, init="custom", update_H=False, H=H)

    W_train_mul_H = W_train @ H
    W_val_mul_H   = W_val   @ H

    ev_train = exp_var(X_train, W_train_mul_H)
    cs_train = cos_sim(X_train, W_train_mul_H)
    ev_val   = exp_var(X_val,   W_val_mul_H)
    cs_val   = cos_sim(X_val,   W_val_mul_H)

    return {
        "seed":          seed,
        "exp_var_train": ev_train,
        "cos_sim_train": cs_train,
        "exp_var_val":   ev_val,
        "cos_sim_val":   cs_val,
        "H":             H,
        "W_train":       W_train,
        "W_val":         W_val
    }


def run_nmf_loop(X_train: pd.DataFrame, X_val: pd.DataFrame, n_signatures: int, n_runs: int, n_jobs: int = -2):
    """Train NMF with multiple random seeds, return best model and per-run history.

    NMF is non-convex and can get stuck in local minima — multiple random
    initializations are used to find the best solution. Best is selected by
    validation explained variance and cosine similarity.

    Returns:
        best    — dict with keys: exp_var, cos_sim, model, H, W
        history — list of per-run dicts with train/val metrics
    """

    results = Parallel(n_jobs=n_jobs)(delayed(nmf_job)(X_train, X_val, n_signatures, seed) for seed in range(1, n_runs + 1))

    best = {
        "exp_var": -np.inf,
        "cos_sim": -np.inf,
        "H":       None,
        "W":       None,
        "W_val":   None,
    }
    for result in results:        
        if (
            result["exp_var_val"] > best["exp_var"] and 
            result["cos_sim_val"] > best["cos_sim"] and
            not np.isnan(result["exp_var_val"]) and
            not np.isnan(result["cos_sim_val"]) and
            not np.isinf(result["exp_var_val"]) and
            not np.isinf(result["cos_sim_val"])
        ):
            best["exp_var"] = result["exp_var_val"]
            best["cos_sim"] = result["cos_sim_val"]
            best["cos_sim"] = result["cos_sim_val"]
            best["H"]       = result["H"]
            best["W"]       = result["W_train"]
            best["W_val"]   = result["W_val"]

    history = [{
        "seed":          result["seed"],
        "exp_var_train": result["exp_var_train"],
        "cos_sim_train": result["cos_sim_train"],
        "exp_var_val":   result["exp_var_val"],
        "cos_sim_val":   result["cos_sim_val"],
    } for result in results]

    log.info(" Best run | exp_var_val: %.4f | cos_sim_val: %.4f", best["exp_var"], best["cos_sim"])
    return best, history


