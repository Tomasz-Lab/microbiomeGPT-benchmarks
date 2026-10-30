import logging

from gut_former.utils.metrics import cos_sim, exp_var

log = logging.getLogger(__name__)


def test_nmf_reconstructs_X(nmf_model, synthetic_nmf):
    X = synthetic_nmf.X
    model = nmf_model.fit(X)
    X_pred = model.W_train_ @ model.H_

    exp_var_score = exp_var(X, X_pred)
    cos_sim_val_score = cos_sim(X, X_pred)
    log.debug(f"exp_var={exp_var_score} cos_sim={cos_sim_val_score}")

    assert exp_var_score > 0.99
    assert cos_sim_val_score > 0.99


def test_transform_on_unseen_samples(nmf_model, synthetic_split):
    model = nmf_model.fit(synthetic_split.X_train)
    res = model.evaluate(synthetic_split.X_val)

    exp_var_score = res["exp_var_val"]
    cos_sim_val_score = res["cos_sim_val"]

    log.debug(f"exp_var={exp_var_score} cos_sim={cos_sim_val_score}")

    assert res["W_val"].shape == (len(synthetic_split.X_val), synthetic_split.k)
    assert (res["W_val"] >= 0).all()
    assert res["exp_var_val"] > 0.98
    assert res["cos_sim_val"] > 0.98
