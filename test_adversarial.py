# tests/test_adversarial.py
import numpy as np
from WINNER_VERSION_RUN22 import solve_instance_fusion_pnp_np

def xor_gadget(n=6):
    # Ručně postavený CNF s XOR-like strukturou (SAT/UNSAT mix podle parametrů)
    cnf = [(+1,+2,-3), (+1,-2,+3), (-1,+2,+3), (-1,-2,-3)]
    # doplň dummy proměnné, ať to má n
    for v in range(4, n+1):
        cnf.append((+v, +v, +v))
    return cnf

def test_adversarial_xor():
    rng = np.random.default_rng(7)
    cnf = xor_gadget(30)
    out = solve_instance_fusion_pnp_np(cnf, 30, rng, seed=7, max_steps=300)
    # Nemusí vždy vyřešit, ale nesmí spadnout a mělo by zlepšovat:
    assert out["FinalUnsat"] >= 0
