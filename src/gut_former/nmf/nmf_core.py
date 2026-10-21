import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import non_negative_factorization

from gut_former.utils.metrics import cos_sim, exp_var

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NMFFitResult:
    seed: int
    exp_var: float
    cos_sim: float
    W: np.ndarray
    H: np.ndarray


class NMF(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components: int,
        init: str,
        solver: str,
        beta_loss: str,
        max_iter: int,
        n_runs: int,
        n_jobs: int,
        selection_metric: str,
    ):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.max_iter = max_iter
        self.n_runs = n_runs
        self.n_jobs = n_jobs
        self.selection_metric = selection_metric

        self.H_: np.ndarray | None = None
        self.W_train_: np.ndarray | None = None
        self.best_seed_: int | None = None
        self.history_: list[dict[str, float]] = []

    def _single_fit_run(self, X: np.ndarray, seed: int) -> NMFFitResult:
        W, H, _ = non_negative_factorization(
            X,
            n_components=self.n_components,
            init=self.init,
            solver=self.solver,
            beta_loss=self.beta_loss,
            max_iter=self.max_iter,
            random_state=seed,
        )

        if np.isnan(W).any() or np.isnan(H).any():
            log.warning("NMF run (seed %d) returned NaN values.", seed)
            return NMFFitResult(
                seed=seed,
                exp_var=np.nan,
                cos_sim=np.nan,
                W=W,
                H=H,
            )

        pred = W @ H

        return NMFFitResult(
            seed=seed,
            exp_var=exp_var(X, pred),
            cos_sim=cos_sim(X, pred),
            W=W,
            H=H,
        )

    def _get_metric(self, run: NMFFitResult) -> float:
        if self.selection_metric == "exp_var":
            return run.exp_var
        elif self.selection_metric == "cos_sim":
            return run.cos_sim
        elif self.selection_metric == "composite":
            return run.exp_var + run.cos_sim
        raise ValueError(f"Unknown selection metric: {self.selection_metric}")

    def fit(self, X: pd.DataFrame | np.ndarray, y=None):
        X_arr = X.values if isinstance(X, pd.DataFrame) else X

        results: list[NMFFitResult] = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_fit_run)(X_arr, seed) for seed in range(1, self.n_runs + 1)
        )

        valid_results = [r for r in results if np.isfinite(self._get_metric(r))]

        if not valid_results:
            raise RuntimeError("All NMF runs returned invalid values (NaN/Inf).")

        best_run = max(valid_results, key=self._get_metric)

        self.H_ = best_run.H
        self.W_train_ = best_run.W
        self.best_seed_ = best_run.seed

        self.history_ = [
            {
                "seed": r.seed,
                "exp_var_train": r.exp_var,
                "cos_sim_train": r.cos_sim,
            }
            for r in results
        ]

        log.info(
            "Best run (seed %d) | exp_var_train: %.4f | cos_sim_train: %.4f",
            best_run.seed,
            best_run.exp_var,
            best_run.cos_sim,
        )
        return self

    def transform(self, X, H: np.ndarray | None = None) -> np.ndarray:
        H = self.H_ if H is None else H
        if H is None:
            raise RuntimeError("Model is not yet fitted. Call .fit() first.")

        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        W, _, _ = non_negative_factorization(
            X_arr,
            n_components=self.n_components,
            init="custom",
            solver=self.solver,
            beta_loss=self.beta_loss,
            max_iter=self.max_iter,
            update_H=False,
            H=H,
        )
        return W

    def evaluate(self, X_val: pd.DataFrame | np.ndarray) -> dict[str, float]:
        W_val = self.transform(X_val)
        X_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val

        reconstruction = W_val @ self.H_

        return {
            "W_val": W_val,
            "exp_var_val": exp_var(X_arr, reconstruction),
            "cos_sim_val": cos_sim(X_arr, reconstruction),
        }
