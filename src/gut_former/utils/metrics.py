import numpy as np

from gut_former.utils.types import Matrix


def cos_sim(
    X: Matrix,
    X_pred: Matrix,
) -> float:
    X_flat = np.concatenate(np.array(X))
    X_pred_flat = np.concatenate(np.array(X_pred))
    return float(
        X_pred_flat.dot(X_flat) / np.sqrt(X_pred_flat.dot(X_pred_flat) * X_flat.dot(X_flat))
    )


def exp_var(
    X: Matrix,
    X_pred: Matrix,
) -> float:
    return float(1 - (np.sum((np.array(X) - np.array(X_pred)) ** 2) / np.sum(np.array(X) ** 2)))


def rss(
    X: Matrix,
    X_pred: Matrix,
) -> np.float64:
    return np.float64(np.sum((np.array(X) - np.array(X_pred)) ** 2))


def l2norm(
    X: Matrix,
    X_pred: Matrix,
) -> float:
    return float(np.sqrt(np.sum((np.array(X) - np.array(X_pred)) ** 2)))
