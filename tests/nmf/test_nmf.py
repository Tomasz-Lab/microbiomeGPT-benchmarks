import logging

from gut_former.utils.metrics import cos_sim, exp_var

log = logging.getLogger(__name__)


def test_nmf_reconstructs_X(fitted_nmf_model, synthetic_nmf):
    X = synthetic_nmf.X
    X_pred = fitted_nmf_model.W_train_ @ fitted_nmf_model.H_

    ev = exp_var(X, X_pred)
    cs = cos_sim(X, X_pred)
    log.debug("exp_var=%.4f cos_sim=%.4f", ev, cs)

    assert ev > 0.99
    assert cs > 0.99
