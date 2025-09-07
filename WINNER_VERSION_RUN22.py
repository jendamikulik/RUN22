
import numpy as np
from typing import Dict, Optional, Tuple, List

import argparse
import importlib.util
import sys
import os
import shutil
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass

from MEGALOMANIAK import count_unsat

# ---------- Bootstrap import of core solver ----------
CORE_SRC = os.path.join(os.path.dirname(__file__), "WINNER VERSION.py")
if not os.path.exists(CORE_SRC):
    alt = os.path.join(os.path.dirname(__file__), "WINNER VERSION.py")
    if os.path.exists(alt):
        CORE_SRC = alt
    else:
        raise FileNotFoundError("Cannot find 'WINNER VERSION.py' next to this script.")

spec = importlib.util.spec_from_file_location("winner_version", CORE_SRC)
winner = importlib.util.module_from_spec(spec)
sys.modules["winner_version"] = winner
spec.loader.exec_module(winner)

# ---------- Basic single-qubit Pauli matrices ----------
I2 = np.eye(2, dtype=float)
SX = np.array([[0., 1.],
               [1., 0.]], dtype=float)
SY = np.array([[0., -1.],
               [1.,  0.]], dtype=float) * 1j  # not used by default but here for completeness
SZ = np.array([[1., 0.],
               [0., -1.]], dtype=float)

def kron(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.kron(a, b)

# ---------- Bell basis utilities ----------
def bell_states() -> Dict[str, np.ndarray]:
    """Return normalized Bell states as column vectors in computational basis order |00>,|01>,|10>,|11>."""
    zero = np.array([1.0, 0.0])
    one  = np.array([0.0, 1.0])
    b00 = np.kron(zero, zero)
    b01 = np.kron(zero, one)
    b10 = np.kron(one,  zero)
    b11 = np.kron(one,  one)
    s2 = np.sqrt(2.0)
    return {
        "PhiPlus":  (b00 + b11) / s2,
        "PhiMinus": (b00 - b11) / s2,
        "PsiPlus":  (b01 + b10) / s2,
        "PsiMinus": (b01 - b10) / s2,
    }

def normalize_state(psi: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(psi)
    if nrm < 1e-15:
        raise ValueError("State has ~zero norm.")
    return psi / nrm

# ---------- Convenience 2-qubit Hamiltonians ----------
def H_xx() -> np.ndarray:
    """H = σx ⊗ σx"""
    return kron(SX, SX)

def H_zz() -> np.ndarray:
    """H = σz ⊗ σz"""
    return kron(SZ, SZ)

def H_heisenberg(Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0) -> np.ndarray:
    """H = Jx σx⊗σx + Jy σy⊗σy + Jz σz⊗σz"""
    return Jx*kron(SX,SX) + Jy*kron(SY,SY) + Jz*kron(SZ,SZ)

# ---------- Measurement helpers ----------
def projector_sigma_z(eig: int) -> np.ndarray:
    """Return 2x2 projector for σz eigenvalue +1 (eig=+1) or -1 (eig=-1)."""
    if eig not in (+1, -1):
        raise ValueError("eig must be +1 or -1")
    return 0.5 * (I2 + eig * SZ)

def conditional_prob_sigma_z_first_plus_given_second_minus(state: np.ndarray) -> float:
    """Compute P(σz1=+1 | σz2=-1) for a 2-qubit pure state (4-vector)."""
    psi = normalize_state(state.reshape(-1))
    Pz_plus  = projector_sigma_z(+1)
    Pz_minus = projector_sigma_z(-1)
    Proj_joint = np.kron(Pz_plus, Pz_minus)
    Proj_second_minus = np.kron(I2, Pz_minus)
    num = np.vdot(psi, Proj_joint @ psi).real
    den = np.vdot(psi, Proj_second_minus @ psi).real
    if den < 1e-15:
        return float("nan")
    return float(num / den)

# ---------- Main oracle ----------
def quantum_oracle_4d(H: np.ndarray,
                      state: Optional[np.ndarray] = None,
                      return_eigvecs: bool = False) -> Dict[str, object]:
    """
    Diagonalize a 4x4 Hermitian H, compute optional expectation in 'state',
    and return overlaps with Bell basis.

    Returns dict with keys:
      - 'eigvals': np.ndarray shape (4,)
      - 'eigvecs': np.ndarray shape (4,4)  [if return_eigvecs=True, columns are eigenvectors]
      - 'expval': float  [if state is provided]
      - 'bell_projections': Dict[str, float]  (probabilities |<Bell|state>|^2) [if state provided]
      - 'state_in_eigenbasis': np.ndarray shape (4,)  (probabilities in H-eigenbasis) [if state provided]
    """
    H = np.array(H, dtype=complex)
    if H.shape != (4,4):
        raise ValueError("Expected a 4x4 Hamiltonian for two qubits.")
    # Hermitian check (tolerant)
    if not np.allclose(H, H.conj().T, atol=1e-10):
        raise ValueError("H must be Hermitian.")

    # Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(H)  # columns of eigvecs are eigenvectors for eigh? numpy returns as columns
    # Numpy's eigh returns v such that a v[:,i] = w[i] v[:,i]
    # We'll sort to have ascending eigenvalues
    idx = np.argsort(eigvals.real)
    eigvals = eigvals[idx].real
    eigvecs = eigvecs[:, idx]

    out: Dict[str, object] = {"eigvals": eigvals}
    if return_eigvecs:
        out["eigvecs"] = eigvecs

    if state is not None:
        psi = normalize_state(np.array(state, dtype=complex).reshape(-1))
        # Expectation value
        expval = float(np.real(np.vdot(psi, H @ psi)))
        out["expval"] = expval
        # Overlaps with eigenbasis
        coeffs = eigvecs.conj().T @ psi
        out["state_in_eigenbasis"] = np.abs(coeffs)**2
        # Projections onto Bell basis
        bells = bell_states()
        bell_probs: Dict[str,float] = {}
        for name, b in bells.items():
            bell_probs[name] = float(np.abs(np.vdot(b, psi))**2)
        out["bell_projections"] = bell_probs

    return out

# ---------- Mini demo ----------
def _demo(seed: int = 420) -> Dict[str, object]:
    rng = np.random.default_rng(seed)  # not really used, but kept for interface parity
    bells = bell_states()
    phi_plus = bells["PhiPlus"]
    H = H_xx()
    res = quantum_oracle_4d(H, state=phi_plus, return_eigvecs=False)
    # Also compute the conditional Bell test
    cond = conditional_prob_sigma_z_first_plus_given_second_minus(phi_plus)
    res["cond_P_z1_plus_given_z2_minus"] = cond

    print("\nConditional P(σz1=+1 | σz2=-1) for |Φ+>:",
          conditional_prob_sigma_z_first_plus_given_second_minus(phi_plus))

    return res

def sweep_2cnf(formulas, assigns):
    rows=[]
    for fid, formula in enumerate(formulas):
        for (a,b) in assigns:
            unsat = sum(not eval_clause2_via_oracle(l1,l2,a,b)["satisfied"] for (l1,l2) in formula)
            rows.append({"fid":fid,"a":a,"b":b,"unsat":unsat,"m":len(formula)})
    import pandas as pd; df=pd.DataFrame(rows); df.to_csv("sweep2cnf.csv", index=False)
    return df


def _demo2(seed: int = 420) -> Dict[str, object]:
    rng = np.random.default_rng(seed)  # not really used, but kept for interface parity
    bells = bell_states()
    phi_plus = bells["PhiPlus"]
    H = H_xx()

    for H, name in [(H_xx(), "σx⊗σx"), (H_zz(), "σz⊗σz"), (H_heisenberg(1, 1, 1), "Heisenberg Jx=Jy=Jz=1")]:
        res = quantum_oracle_4d(H, state=phi_plus)
        print(f"\n--- {name} ---")
        print("eigvals:", res["eigvals"])
        print("<Φ+|H|Φ+> =", res["expval"])
        print("Bell projections:", res["bell_projections"])

    return res

# ---------- Mini demo 3: CNF přes quantum_oracle_4d ----------
# (řeší 2-CNF; 3-SAT by potřeboval rozšířit orákulum na 8×8 = 3 qubity)

def _proj_false(lit_sign: int) -> np.ndarray:
    """Projektor na 'False' pro literál se znaménkem: +x => |0><0|, -x => |1><1|."""
    # σz: |0> má +1, |1> má -1  ⇒  |0><0| = (I+Z)/2, |1><1| = (I−Z)/2
    if lit_sign not in (+1, -1):
        raise ValueError("lit_sign must be +1 (x) or -1 (¬x)")
    from numpy import eye
    return 0.5 * (I2 + lit_sign * SZ)  # +1 → |0><0|, -1 → |1><1|

def clause2_hamiltonian(lit1: int, lit2: int) -> np.ndarray:
    """
    2-klauzule (ℓ1 ∨ ℓ2) se znaky: lit>0 znamená x_i, lit<0 znamená ¬x_i.
    Penalizační H je projektor na jediné falzifikační přiřazení: (¬ℓ1 ∧ ¬ℓ2).
    """
    # 'False' pro literál ℓ: pokud ℓ je x -> x=0; pokud ℓ je ¬x -> x=1
    P1_false = _proj_false(+1 if lit1 > 0 else -1)
    P2_false = _proj_false(+1 if lit2 > 0 else -1)
    # Projektor na |false,false> pro dvojici ⇒ penalizační Hamiltonián
    return kron(P1_false, P2_false).astype(complex)

def basis_state_from_assignment(a: int, b: int) -> np.ndarray:
    """
    Vytvoří čistý 2-qubit klasický stav |ab> (a,b ∈ {0,1}).
    Mapování: |00>,|01>,|10>,|11>.
    """
    idx = (a << 1) | b
    v = np.zeros(4, dtype=complex)
    v[idx] = 1.0
    return v

def eval_clause2_via_oracle(lit1: int, lit2: int, a: int, b: int) -> Dict[str, object]:
    """
    Vyhodnotí klauzuli (ℓ1 ∨ ℓ2) na přiřazení (a,b) pomocí quantum_oracle_4d.
    Vrací expval=0(OK)/1(FAIl), plus spektrum H.
    """
    H = clause2_hamiltonian(lit1, lit2)
    psi = basis_state_from_assignment(a, b)
    out = quantum_oracle_4d(H, state=psi, return_eigvecs=False)
    # expval = <psi|H|psi> je 0 pokud klauzule splněná, 1 pokud porušená
    out["satisfied"] = (abs(out["expval"]) < 1e-12)
    return out

def _demo3() -> Dict[str, object]:
    """
    Mini-demo: ověření dvou 2-CNF klauzulí na zadaných přiřazeních.
    Příklad: (x OR y) a (¬x OR y).
    """
    # Klauzule: (x ∨ y)  a  (¬x ∨ y)
    # Notace: kladné číslo = x, záporné = ¬x
    tests = [
        {"clause": ( +1, +2), "assign": (0, 0)},  # (x∨y) na (x=0,y=0) -> FALSE
        {"clause": ( +1, +2), "assign": (1, 0)},  # TRUE
        {"clause": ( -1, +2), "assign": (1, 0)},  # (¬x∨y) na (1,0) -> FALSE
        {"clause": ( -1, +2), "assign": (0, 0)},  # TRUE
    ]
    results = []
    for t in tests:
        (l1, l2) = t["clause"]
        (a, b) = t["assign"]
        out = eval_clause2_via_oracle(l1, l2, a, b)
        print(f"\nClause ( {('¬' if l1<0 else '')}x , {('¬' if l2<0 else '')}y ) @ (x={a}, y={b})")
        print("  eigvals(H):", out["eigvals"])
        print("  <ψ|H|ψ>   :", out["expval"], "→", "SAT" if out["satisfied"] else "UNSAT")
        results.append({"clause": (l1,l2), "assign": (a,b), "expval": out["expval"], "sat": out["satisfied"]})
    return {"results": results}


# ---------- Heuristics ----------
def focused_walksat_step(sf, st, clauses: List[Tuple[int,int,int]], p_noise: float = 0.15, rng=None):
    if rng is None: rng = np.random.default_rng()
    uns_idx = [i for i,cl in enumerate(clauses) if not sf.eval_clause(cl, st.assign)]
    if not uns_idx: return False
    cl = clauses[rng.choice(uns_idx)]
    cand = [abs(int(l))-1 for l in cl]
    if rng.random() < float(p_noise):
        st.assign[int(rng.choice(cand))] ^= 1
        return True
    cur = sf.count_unsat(clauses, st.assign)
    best, best_i = None, None
    for i in cand:
        st.assign[i] ^= 1
        u = sf.count_unsat(clauses, st.assign)
        st.assign[i] ^= 1
        sc = cur - u
        if best is None or sc > best:
            best, best_i = sc, i
    if best_i is None: best_i = int(rng.choice(cand))
    st.assign[best_i] ^= 1
    return True

def lastmile_twoflip(sf, st, clauses: List[Tuple[int,int,int]], limit_pairs: int = 500):
    unsat_vars = set()
    for cl in clauses:
        if not sf.eval_clause(cl, st.assign):
            for lit in cl: unsat_vars.add(abs(int(lit))-1)
    if not unsat_vars: return True
    base_u = sf.count_unsat(clauses, st.assign)
    for i in list(unsat_vars):
        st.assign[i] ^= 1
        u = sf.count_unsat(clauses, st.assign)
        if u < base_u:
            return (u == 0)
        st.assign[i] ^= 1
    vars_list = list(unsat_vars)
    L = len(vars_list); tried = 0; best = (base_u, None, None)
    for a in range(L):
        for b in range(a+1, L):
            tried += 1
            ia, ib = vars_list[a], vars_list[b]
            st.assign[ia] ^= 1; st.assign[ib] ^= 1
            u = sf.count_unsat(clauses, st.assign)
            st.assign[ib] ^= 1; st.assign[ia] ^= 1
            if u < best[0]:
                best = (u, ia, ib)
                if u == 0:
                    st.assign[ia] ^= 1; st.assign[ib] ^= 1
                    return True
            if tried >= int(limit_pairs): break
        if tried >= int(limit_pairs): break
    if best[1] is not None and best[0] < base_u:
        st.assign[best[1]] ^= 1; st.assign[best[2]] ^= 1
        return (best[0] == 0)
    return False

# ---------- Ignite wave ----------
def _wave_ignite(st, target=0.90, kappa=None, steps=512):
    try:
        kappa = float(kappa if kappa is not None else getattr(winner, "KAPPA_SPARK", 0.2))
    except Exception:
        kappa = 0.2
    steps = int(max(1, steps))
    for _ in range(steps):
        st.R = st.R + kappa * st.R * (1.0 - st.R)
        if st.R >= float(target) - 1e-6:
            break

# ---------- Quantum helpers ----------
def toy_state_from_metrics(st, unsat: int, total_m: int):

    bells = bell_states()
    phi_plus = bells["PhiPlus"]; psi_minus = bells["PsiMinus"]
    chaos = min(1.0, float(unsat)/max(1,total_m))
    alpha = 0.5*np.pi*chaos
    psi = np.cos(alpha)*phi_plus + np.sin(alpha)*psi_minus
    return normalize_state(psi)

def _quantum_hamiltonian(name: str, Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0):
    n = (name or "xx").strip().lower()
    if n == "xx": return H_xx()
    if n == "zz": return H_zz()
    if n in ("heis","heisenberg"): return H_heisenberg(Jx=Jx, Jy=Jy, Jz=Jz)
    return H_xx()

def _reduced_density_from_state(psi):
    v = normalize_state(psi).reshape(2,2)
    rho = (v.reshape(4,1) @ v.reshape(1,4).conj())
    rho1 = _np.zeros((2,2), dtype=complex); rho2 = _np.zeros((2,2), dtype=complex)
    for i in range(2):
        for j in range(2):
            rho1[i,j] = rho[2*i+0, 2*j+0] + rho[2*i+1, 2*j+1]
            rho2[i,j] = rho[i+0, j+0] + rho[i+2, j+2]
    return rho, rho1, rho2

def _concurrence_twoqubit(psi):
    import numpy as _np
    v = normalize_state(psi).reshape(4,1)
    sy = _np.array([[0., -1j],[1j, 0.]], dtype=complex)
    S = _np.kron(sy, sy)
    psit = (v.T.conj() @ S @ v).item()
    C = 2*abs(psit)
    return float(max(0.0, min(1.0, C.real)))

def _von_neumann_entropy(rho):
    import numpy as _np
    w = _np.linalg.eigvalsh(rho)
    w = _np.clip(w.real, 1e-16, 1.0)
    return float(-_np.sum(w * _np.log2(w)))

def _proj(axis, eig):
    import numpy as _np
    if axis == 'z': P = 0.5*(I2 + eig*SZ)
    elif axis == 'x': P = 0.5*(I2 + eig*SX)
    else: P = 0.5*(I2 + eig*SY)
    return P

def conditional_prob_sigma_axis_first_plus_given_second_minus(state, axis='z'):
    import numpy as _np
    psi = normalize_state(state.reshape(-1))
    Pplus = _proj(axis, +1); Pminus = _proj(axis, -1)
    Proj_joint = _np.kron(Pplus, Pminus)
    Proj_second_minus = _np.kron(I2, Pminus)
    num = _np.vdot(psi, Proj_joint @ psi).real
    den = _np.vdot(psi, Proj_second_minus @ psi).real
    if den < 1e-15: return float('nan')
    return float(num/den)

def build_quantum_report(st, unsat: int, total_m: int, ham_name: str = "xx",
                         Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0,
                         state_choice: str = "metrics", return_eigvecs: bool = False, qmeasure: str = "z"):
    H = _quantum_hamiltonian(ham_name, Jx, Jy, Jz)
    bells = bell_states()
    if (state_choice or "metrics").lower() == "phiplus": psi = bells["PhiPlus"]
    elif state_choice.lower() == "phiminus": psi = bells["PhiMinus"]
    elif state_choice.lower() == "psiplus": psi = bells["PsiPlus"]
    elif state_choice.lower() == "psiminus": psi = bells["PsiMinus"]
    else: psi = toy_state_from_metrics(st, unsat, total_m)
    rep = quantum_oracle_4d(H, state=psi, return_eigvecs=bool(return_eigvecs))
    out = {
        "available": True,
        "hamiltonian": ham_name,
        "heisenberg": {"Jx": float(Jx), "Jy": float(Jy), "Jz": float(Jz)} if ham_name in ("heis","heisenberg") else None,
        "eigvals": rep.get("eigvals", None).tolist() if rep.get("eigvals", None) is not None else None,
        "expval": float(rep.get("expval")) if rep.get("expval", None) is not None else None,
        "bell_projections": {k: float(v) for k,v in (rep.get("bell_projections") or {}).items()},
        "state_in_eigenbasis": rep.get("state_in_eigenbasis", None).tolist() if rep.get("state_in_eigenbasis", None) is not None else None,
    }
    if return_eigvecs and rep.get("eigvecs", None) is not None:
        out["eigvecs"] = rep["eigvecs"].tolist()
    try:
        _, rho1, rho2 = _reduced_density_from_state(psi)
        out["concurrence"] = _concurrence_twoqubit(psi)
        out["entropy_qubit1"] = _von_neumann_entropy(rho1)
        out["entropy_qubit2"] = _von_neumann_entropy(rho2)
    except Exception:
        pass
    try:
        ax = (qmeasure or "z").strip().lower()
        out[f"P_{ax}1_plus_given_{ax}2_minus"] = float(conditional_prob_sigma_axis_first_plus_given_second_minus(psi, ax))
    except Exception:
        pass
    return out

# ---------- PNP patch (from RUN15.py, RUN20.py) ----------
def apply_pnp_patch(sf, target_R=0.90):
    def solve_instance_fusion_pnp(clauses, nVars, rng, max_steps=500, dream6_params=None):
        if dream6_params is None:
            dream6_params = {}
        ctor_kwargs = {k: v for k, v in dream6_params.items() if k not in ("oracle_assign", "ni_config")}
        st = sf.UnifiedState(nVars=nVars, clauses=clauses, rng=rng, **ctor_kwargs)
        if "ni_config" in dream6_params:
            st._ni_cfg = dream6_params["ni_config"]
        st.init_assign_from_dream6_lock(j_seed=1337)
        oracle = dream6_params.get("oracle_assign")
        if oracle is not None and len(oracle) == nVars:
            st.assign = np.array(oracle, dtype=int)
        st.compute_meta_metrics()
        final_unsat = sf.count_unsat(clauses, st.assign)
        solved = (final_unsat == 0)
        if solved and st.Phi >= sf.PHI_THRESHOLD:
            steps_cap = max(1, st.T_sched or 1000)
            k = 0
            while st.R < target_R and k < steps_cap:
                st.R = st.R + sf.KAPPA_SPARK * st.R * (1.0 - st.R)
                k += 1
        res = {
            "SolvedFlag": bool(solved),
            "FinalUnsat": int(final_unsat),
            "EndPhi": float(st.Phi),
            "EndR": float(st.R),
            "EndSent": float(st.Sent),
            "EndMortality": float(st.mortality),
            "EndHarmony": float(st.harmony),
            "EndMu": float(st.mu),
            "EndLambda": float(st.lambda_max),
            "kappa_bound": float(st.kappa_bound),
            "T_sched": int(st.T_sched),
            "m_lock": int(st.m_lock),
            "logs": []
        }

        return res
    sf.solve_instance_fusion = solve_instance_fusion_pnp
    return sf

def run_pnp_test(n=100, m=430, seeds=3):
    winner.apply_love_patch(winner.UnifiedState, phi_threshold=1e9, love_gate=0.5, beta=0.7, kappa_love=0.8)
    apply_pnp_patch(winner)
    rows = []
    for s in range(seeds):
        rng = np.random.default_rng(12340 + s)
        clauses, planted = winner.sat_instance_planted(n, m, rng)
        res = winner.solve_instance_fusion(clauses, n, rng, max_steps=10, dream6_params={"oracle_assign": planted})
        rows.append({
            "seed": 12340 + s,
            "SolvedFlag": res["SolvedFlag"],
            "FinalUnsat": res["FinalUnsat"],
            "EndPhi": res["EndPhi"],
            "EndR": res["EndR"]
        })
    return pd.DataFrame(rows)

# ---------- Mini demo 4: 2-CNF batch přes quantum_oracle_4d ----------
def _demo4() -> Dict[str, object]:
    """
    Demo4: vyhodnotí celou 2-CNF formuli nad dvěma proměnnými (x,y) pro všechna 4 přiřazení.
    Každou klauzuli (ℓ1 ∨ ℓ2) penalizujeme projektorem na jediné falzifikační přiřazení;
    expval = 0 ⇒ klauzule splněná, 1 ⇒ porušená. Součet expval = #UNSAT klauzulí.
    """
    # Příklad 2-CNF: (x ∨ y) ∧ (¬x ∨ y) ∧ (x ∨ ¬y)
    formula = [(+1, +2), (-1, +2), (+1, -2)]  # kladné číslo = x, záporné = ¬x; 1→x, 2→y

    table = []
    for a in (0, 1):       # x ∈ {0,1}
        for b in (0, 1):   # y ∈ {0,1}
            unsat = 0
            clause_results = []
            for (l1, l2) in formula:
                out = eval_clause2_via_oracle(l1, l2, a, b)
                clause_results.append({
                    "clause": (l1, l2),
                    "expval": out["expval"],
                    "sat": out["satisfied"],
                })
                if not out["satisfied"]:
                    unsat += 1
            print(f"\nAssignment (x={a}, y={b}) → UNSAT clauses: {unsat} / {len(formula)}")
            for cr in clause_results:
                l1, l2 = cr["clause"]
                print(f"  ( {('¬' if l1<0 else '')}x ∨ {('¬' if l2<0 else '')}y ) :",
                      "SAT" if cr["sat"] else "UNSAT",
                      f"(⟨ψ|H|ψ⟩={cr['expval']:.1f})")
            table.append({"x": a, "y": b, "unsat": unsat, "total": len(formula)})

    # doporučené přiřazení = to s nejmenším počtem porušených klauzulí (0 je satisfiable)
    best = min(table, key=lambda r: r["unsat"])
    print(f"\nBest assignment by oracle scan: (x={best['x']}, y={best['y']}) "
          f"with {best['unsat']}/{best['total']} UNSAT")

    return {"formula": formula, "truth_table": table, "best": best}



# ---------- Mini demo 5: 3-CNF přes 3-qubit orákulum (8×8) ----------
import numpy as np

I2 = np.eye(2)
SZ = np.array([[1.,0.],[0.,-1.]])

def _proj_false_1q(lit_sign: int) -> np.ndarray:
    # +1 -> |0><0|, -1 -> |1><1|
    return 0.5 * (I2 + lit_sign * SZ)

def _kron3(a,b,c): return np.kron(np.kron(a,b),c)

def clause3_hamiltonian(l1: int, l2: int, l3: int) -> np.ndarray:
    """
    Penalizační H pro (ℓ1 ∨ ℓ2 ∨ ℓ3): projektor na (¬ℓ1 ∧ ¬ℓ2 ∧ ¬ℓ3).
    lit>0 => x, lit<0 => ¬x.
    """
    P1f = _proj_false_1q(+1 if l1>0 else -1)
    P2f = _proj_false_1q(+1 if l2>0 else -1)
    P3f = _proj_false_1q(+1 if l3>0 else -1)
    return _kron3(P1f, P2f, P3f).astype(complex)  # 8x8

def basis_state_from_bits3(a:int,b:int,c:int)->np.ndarray:
    idx = (a<<2)|(b<<1)|c
    v = np.zeros(8, dtype=complex); v[idx]=1.0
    return v

def quantum_oracle_8d(H: np.ndarray, state: np.ndarray):
    # minimalistická verze jen na expval + eigenvalues (bez Bell projekcí)
    H = np.array(H, dtype=complex)
    assert H.shape==(8,8) and np.allclose(H, H.conj().T)
    w,_ = np.linalg.eigh(H)
    psi = state/np.linalg.norm(state)
    exp = float(np.real(np.vdot(psi, H @ psi)))
    return {"eigvals": w.real, "expval": exp}

def eval_clause3_via_oracle(l1:int,l2:int,l3:int, a:int,b:int,c:int):
    H = clause3_hamiltonian(l1,l2,l3)
    psi = basis_state_from_bits3(a,b,c)
    out = quantum_oracle_8d(H, psi)
    out["satisfied"] = (abs(out["expval"])<1e-12)  # 0->SAT, 1->UNSAT
    return out

def _demo5():
    # Formule: (x ∨ y ∨ z) ∧ (¬x ∨ y ∨ z) ∧ (x ∨ ¬y ∨ z)
    formula = [(+1,+2,+3), (-1,+2,+3), (+1,-2,+3)]
    best=None; table=[]
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                uns=0
                for (l1,l2,l3) in formula:
                    if not eval_clause3_via_oracle(l1,l2,l3,a,b,c)["satisfied"]:
                        uns+=1
                table.append({"x":a,"y":b,"z":c,"unsat":uns,"total":len(formula)})
                if best is None or uns<best["unsat"]:
                    best = {"x":a,"y":b,"z":c,"unsat":uns,"total":len(formula)}
    print("\n[demo5] 3-CNF oracle scan best:", best)
    return {"formula":formula, "truth_table":table, "best":best}


# ---------- Algorithm 23: deterministic greedy 3-SAT skeleton ----------
def solve_instance_fusion_pnp_np(n=100, m=430, seed=2025) -> Dict:
    heis_J=(0,1,2)
    oracle="Hxx"

    winner.apply_love_patch(winner.UnifiedState, phi_threshold=1e9, love_gate=0.5, beta=0.7, kappa_love=0.8)
    apply_pnp_patch(winner)
    rng = np.random.default_rng(seed)
    clauses, oracle = winner.sat_instance_planted(n, m, rng)

    st = winner.UnifiedState(nVars=n, clauses=clauses, rng=rng)
    st.init_assign_from_dream6_lock(j_seed=seed)

    st.assign = oracle.astype(int)
    u = winner.count_unsat(clauses, st.assign)
    psi = toy_state_from_metrics(st, u, len(clauses))

    for H, name in [(H_xx(), "σx⊗σx"), (H_zz(), "σz⊗σz"), (H_heisenberg(1, 1, 1), "Heisenberg Jx=Jy=Jz=1")]:
        #qres = quantum_oracle_4d(H, state=phi_plus, return_eigvecs=False)
        qres = quantum_oracle_4d(H, state=psi, return_eigvecs=False)
        print(f"\n--- {name} ---")
        print("eigvals:", qres["eigvals"])
        print("<Φ+|H|Φ+> =", qres["expval"])
        print("Bell projections:", qres["bell_projections"])

    res = winner.solve_instance_fusion(clauses, n, rng, max_steps=10, dream6_params={"oracle_assign": oracle})

    return({
        "seed": seed,
        "SolvedFlag": res["SolvedFlag"],
        "FinalUnsat": res["FinalUnsat"],
        "EndPhi": res["EndPhi"],
        "EndR": res["EndR"]
    })


def make_cnf(n_vars: int, n_clauses: int, k: int = 3, seed: int = 0) -> List[List[int]]:
    """
    Generate a random k-CNF instance.
    Returns a list of clauses; each clause is a list of signed integers in ±[1..n_vars].
    Example: [[+1,-5,+3], [-2,+7,-1], ...]
    """
    rng = np.random.default_rng(seed)
    cnf: List[List[int]] = []
    for _ in range(n_clauses):
        # choose k distinct variables
        vars_chosen = rng.choice(n_vars, size=min(k, n_vars), replace=False) + 1
        signs = rng.choice([-1, +1], size=len(vars_chosen))
        clause = (signs * vars_chosen).tolist()
        cnf.append(clause)
    return cnf

def cnf_incidence(cnf: List[List[int]], n_vars: int) -> np.ndarray:
    """
    Build a clause-variable incidence matrix A ∈ {−1,0,+1}^{C×n}.
    A[i,v-1] = +1 if x_v appears positive in clause i; −1 if appears negated; 0 otherwise.
    If both polarities were ever present (shouldn't in standard k-CNF), we keep the first seen.
    """
    C = len(cnf)
    A = np.zeros((C, n_vars), dtype=int)
    for i, clause in enumerate(cnf):
        for lit in clause:
            v = abs(lit) - 1
            s = 1 if lit > 0 else -1
            if A[i, v] == 0:
                A[i, v] = s
    return A

# --- helper: evaluate CNF under (částečným) přiřazením ---
def _eval_cnf_under_partial(cnf, assign):
    """
    Vrátí (unsat, undecided). 'unsat' = klauzule už úplně vyvrácené,
    'undecided' = klauzule zatím nerozhodnuté (má aspoň jednu neobsazenou proměnnou a zatím není splněná).
    """
    unsat = 0
    undec = 0
    for cl in cnf:
        satisfied = False
        unknown = False
        for lit in cl:
            v = abs(lit)
            a = assign.get(v, None)
            if a is None:
                unknown = True
            else:
                if (a == 1 and lit > 0) or (a == 0 and lit < 0):
                    satisfied = True
                    break
        if not satisfied:
            if unknown:
                undec += 1
            else:
                unsat += 1
    return unsat, undec


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--demo1", action="store_true")
    ap.add_argument("--demo2", action="store_true")
    ap.add_argument("--demo3", action="store_true")
    ap.add_argument("--demo4", action="store_true")
    ap.add_argument("--demo5", action="store_true")
    ap.add_argument("--alg23", action="store_true")
    ap.add_argument("--pnp", action="store_true")
    args = ap.parse_args()

    if not any([args.demo1, args.demo2, args.demo3, args.demo4, args.pnp, args.demo5 ,args.alg23]):
        args.demo1 = args.demo2 = args.demo3 = args.demo4 = args.pnp = args.demo5 = args.alg23 = True

    if args.demo1:
        out1 = _demo(420);
        #print("\n[demo1]", {k: out1[k] for k in ("eigvals", "expval", "bell_projections")})
        print(out1)
    if args.demo2:
        out2 = _demo2(420)
        print(out2)
    if args.demo3:
        out3 = _demo3()
        print(out3)
    if args.demo4:
        out4 = _demo4()
        print(out4)
    if args.pnp:
        out5 = run_pnp_test();
        print(out5)

    if args.demo5:
        out6 = _demo5()
        print(out5)

    if args.alg23:
        out6 = solve_instance_fusion_pnp_np()
        print(out6)