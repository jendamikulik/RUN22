# tests/test_oracle_vs_heuristic.py
import numpy as np
import time
import pytest
import winner_version as winner
from WINNER_VERSION_RUN22 import solve_instance_fusion_pnp_np, make_cnf

@pytest.mark.parametrize("n,m", [(60,260), (120,516)])
def test_planted_oracle_mode(n, m):
    rng = np.random.default_rng(1234)
    clauses, planted = winner.sat_instance_planted(n, m, rng)
    out = solve_instance_fusion_pnp_np(clauses, n, rng, seed=2025, max_steps=60,
                                       dream6_params={"oracle_assign": planted})
    assert out["SolvedFlag"] is True
    assert out["FinalUnsat"] == 0

@pytest.mark.parametrize("n,m", [(60,260), (120,516)])
def test_heuristic_mode_success_rate(n, m):
    rng = np.random.default_rng(2025)
    trials, solved = 20, 0
    t0 = time.perf_counter()
    for s in range(trials):
        cnf = make_cnf(n, m, k=3, seed=1000+s)
        out = solve_instance_fusion_pnp_np(cnf, n, rng, seed=1337, max_steps=max(60,int(n*np.log2(n))))
        solved += int(out["SolvedFlag"] and out["FinalUnsat"] == 0)
    dt = time.perf_counter()-t0
    rate = solved/trials
    # nastav si cílové prahy podle reality na tvém stroji
    assert rate >= (0.85 if n==60 else 0.60)
    assert dt < (5.0 if n==60 else 20.0)
