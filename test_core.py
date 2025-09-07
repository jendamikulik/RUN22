# tests/test_core.py
import numpy as np
import pytest
from WINNER_VERSION_RUN22 import (
    eval_clause, count_unsat, make_cnf, cnf_incidence,
    quantum_oracle_4d, H_xx, H_zz, H_heisenberg, bell_states
)

def test_eval_clause_basic():
    cl = ( +1, -2, +3 )  # (x1 ∨ ¬x2 ∨ x3)
    assert eval_clause(cl, np.array([1,0,0], dtype=np.uint8)) is True
    assert eval_clause(cl, np.array([0,1,0], dtype=np.uint8)) is False

def test_count_unsat():
    cnf = [(+1,-2,+3), (-1,+2,-3)]
    a = np.array([1,1,0], dtype=np.uint8)
    assert count_unsat(cnf, a) in (0,1,2)

def test_incidence():
    cnf = [(+1,-2,+3), (-1,+2,-3)]
    A = cnf_incidence(cnf, 3)
    assert set(np.unique(A)).issubset({-1,0,1})
    assert A.shape == (2,3)

def test_quantum_spectra():
    for H in (H_xx(), H_zz()):
        w = np.linalg.eigvalsh(H)
        assert np.allclose(np.sort(w), np.array([-1,-1,1,1]))
    w = np.linalg.eigvalsh(H_heisenberg(1,1,1))
    assert np.allclose(np.sort(w), np.array([-3,1,1,1]))

def test_bell_projections():
    bells = bell_states()
    phi_plus = bells["PhiPlus"]
    res = quantum_oracle_4d(H_xx(), state=phi_plus)
    assert pytest.approx(res["expval"], rel=1e-12) == 1.0
    bp = res["bell_projections"]
    assert bp["PhiPlus"] == pytest.approx(1.0)
    assert bp["PhiMinus"] == pytest.approx(0.0)
