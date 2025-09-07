# tests/test_smoke.py
import numpy as np
from WINNER_VERSION_RUN22 import _demo, _demo2, _demo4, _demo5, run_pnp_test

def test_smoke_demos():
    assert isinstance(_demo(), dict)
    assert isinstance(_demo2(), dict)
    out4 = _demo4(); assert out4["best"]["unsat"] == 0
    out5 = _demo5(); assert out5["best"]["unsat"] == 0

def test_pnp_patch_planted():
    df = run_pnp_test(n=60, m=260, seeds=3)
    assert (df["SolvedFlag"] == True).all()
    assert (df["FinalUnsat"] == 0).all()
