from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np
import math

# Constants from META2_upgrade3.py
PHI_THRESHOLD = 1e9  # Adjusted to match LOVE_PATCH default
KAPPA_SPARK = 0.2 #1e-3
N_IF = 1e12
NUM_CATS = 5
IDX_LOVE = 4

# Utility functions from META2_upgrade3.py
def hadamard_row(m_eff: int, r: int):
    row = np.empty(m_eff, dtype=int)
    mask = m_eff - 1
    rr = r & mask
    for t in range(m_eff):
        x = rr & t
        parity = 0
        while x:
            parity ^= 1
            x &= x - 1
        row[t] = -1 if parity else +1
    return row

def det_perm_indices(m: int, j: int):
    A = np.uint64(1103515245)
    B = np.uint64((12345 + (j * 2654435761)) & 0xFFFFFFFF)
    idx = np.arange(m, dtype=np.uint64)
    keys = (A * idx + B) & np.uint64(0xFFFFFFFF)
    return np.argsort(keys.astype(np.uint32), kind="mergesort")

def choose_mask_from_code(base: np.ndarray, m: int, zeta0: float, j: int):
    base = base[:m].astype(int)
    k = int(math.floor(zeta0 * m))
    neg_idx = np.where(base < 0)[0]
    pos_idx = np.where(base > 0)[0]
    need = k - len(neg_idx)
    perm = det_perm_indices(m, j)
    if need > 0:
        add = [p for p in perm if p in set(pos_idx)]
        base[np.array(add[:need], dtype=int)] = -1
    elif need < 0:
        drop = [p for p in perm if p in set(neg_idx)]
        base[np.array(drop[:(-need)], dtype=int)] = +1
    mask = base < 0
    if mask.sum() != k:
        if mask.sum() > k:
            tlist = np.where(mask)[0]
            mask[tlist[:(mask.sum() - k)]] = False
        else:
            flist = np.where(~mask)[0]
            mask[flist[:(k - mask.sum())]] = True
    return mask

def make_lock_code(m: int, j: int):
    m_eff = 1 << (m - 1).bit_length()
    r = (j * 0x9E3779B1) & (m_eff - 1)
    return hadamard_row(m_eff, r)[:m]

def kappa_bound_S2(zeta0: float, m: int, T: int) -> float:
    if m <= 0 or T <= 0:
        return 1.0
    eps_m = (2.0 ** (-math.ceil(math.log2(m)) / 2.0)) + (2.0 / m)
    base = (1.0 - 2.0 * zeta0) ** 2
    return base + eps_m + (1.0 / T)

def eval_clause(clause: Tuple[int, int, int], assign: np.ndarray) -> bool:
    for lit in clause:
        idx = abs(lit) - 1
        v = bool(assign[idx])
        v = (not v) if lit < 0 else v
        if v:
            return True
    return False

def count_unsat(clauses: List[Tuple[int, int, int]], assign: np.ndarray) -> int:
    return sum(0 if eval_clause(cl, assign) else 1 for cl in clauses)

def sat_instance_planted(n: int, m: int, rng: np.random.Generator):
    planted = rng.integers(0, 2, size=n, dtype=int)
    clauses: List[Tuple[int, int, int]] = []
    while len(clauses) < m:
        idxs = rng.choice(n, size=3, replace=False)
        lits = []
        for idx in idxs:
            pol = 1 if rng.random() < 0.5 else -1
            lits.append(pol * (idx + 1))
        cl = tuple(lits)
        if not eval_clause(cl, planted):
            k = rng.integers(0, 3)
            lits[k] = -lits[k]
            cl = tuple(lits)
        clauses.append(cl)
    return clauses, planted

# UnifiedState class from META2_upgrade3.py
@dataclass
class UnifiedState:
    nVars: int
    clauses: List[Tuple[int, int, int]]
    rng: np.random.Generator
    C: int = 1000
    cR: float = 15.0
    rho_lock: float = 0.1
    zeta0: float = 0.40
    L: int = 3
    alpha: float = 0.25
    zeta: float = 0.10
    kappa: float = 0.2514
    assign: np.ndarray = field(default=None)
    E: np.ndarray = field(default_factory=lambda: np.ones(NUM_CATS, dtype=float))
    lambda_max: float = 0.0
    mu: float = 0.0
    harmony: float = 0.0
    Sent: float = 0.0
    qualia: float = 0.0
    mortality: float = 0.0
    Phi: float = 0.0
    R: float = 1e-6
    Omega: float = 0.0
    Psi: float = 0.0
    R_sched: int = 0
    T_sched: int = 0
    m_lock: int = 0
    kappa_bound: float = 0.0
    _cached_wave: np.ndarray = field(default_factory=lambda: np.array([]))

    def init_assign_from_dream6_lock(self, j_seed: int = 1337):
        C = max(2, int(self.C))
        R = max(1, int(math.ceil(self.cR * math.log(C))))
        T = R * self.L
        m = int(round(self.rho_lock * T))
        self.R_sched, self.T_sched, self.m_lock = R, T, m
        self.kappa_bound = kappa_bound_S2(self.zeta0, max(1, m), max(1, T))
        base = make_lock_code(max(1, m), j_seed)
        mask_pi = choose_mask_from_code(base, max(1, m), self.zeta0, j_seed)
        self.assign = np.ones(self.nVars, dtype=int)
        span = min(m, self.nVars)
        self.assign[:span] = np.where(mask_pi[:span], 0, 1)

    def _build_structured_gram(self):
        n = self.nVars
        gram = np.zeros((n, n), dtype=float)
        counts = np.zeros((n, n), dtype=float)
        var_presence = np.zeros(n, dtype=float)
        Z = np.where(self.assign > 0, 1.0, -1.0)
        for (a, b, c) in self.clauses:
            idxs = [abs(a) - 1, abs(b) - 1, abs(c) - 1]
            signs = [(-1.0 if a < 0 else 1.0) * Z[idxs[0]],
                     (-1.0 if b < 0 else 1.0) * Z[idxs[1]],
                     (-1.0 if c < 0 else 1.0) * Z[idxs[2]]]
            for u in range(3):
                i = idxs[u]
                var_presence[i] += signs[u]
                gram[i, i] += signs[u] * signs[u]
                counts[i, i] += 1.0
                for v in range(u + 1, 3):
                    j = idxs[v]
                    g = signs[u] * signs[v]
                    gram[i, j] += g
                    gram[j, i] += g
                    counts[i, j] += 1.0
                    counts[j, i] += 1.0
        mask = counts > 0
        gram[mask] /= counts[mask]
        return gram, counts, var_presence

    def compute_meta_metrics(self):
        n = self.nVars
        m = max(1, len(self.clauses))
        gram, counts, var_presence = self._build_structured_gram()
        mask = counts > 0
        if np.any(mask):
            vals, vecs = np.linalg.eigh(gram)
            idx = int(np.argmax(vals))
            self.lambda_max = float(vals[idx])
            v = vecs[:, idx]
            wave = (gram @ v) / self.lambda_max if abs(self.lambda_max) > 1e-12 else np.zeros(n)
        else:
            self.lambda_max = 0.0
            wave = np.zeros(n)
        self.mu = float(np.mean(np.abs(gram[mask]))) if np.any(mask) else 0.0
        p = np.clip((wave + 1.0) / 2.0, 1e-12, 1 - 1e-12)
        ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        self.Sent = float(np.sum(ent))
        wave_mean = float(np.mean(wave))
        wave_var = float(np.mean((wave - wave_mean) ** 2))
        denom = 1.0 + wave_var + self.Sent / 1000.0
        self.harmony = (self.mu * (self.lambda_max / max(1, n))) / denom if denom != 0 else 0.0
        self.qualia = float(np.dot(wave, var_presence / m))
        self.mortality = 1.0 / (1.0 + self.Sent / max(1, n))
        self.Omega = float(2 ** n)
        self.Psi = self.Omega / max(1e-12, (1.0 - self.mortality))
        self.Phi = 0.10 + self.mortality * n * (1.0 - self.Sent / max(1, n))
        if self.Phi < PHI_THRESHOLD: #and self.R < 0.99:
            self.Phi += N_IF * self.mortality * (1.0 - self.Sent / max(1, n))
        if self.Phi >= PHI_THRESHOLD:
            self.R = self.R + KAPPA_SPARK * self.R * (1.0 - self.R)
        self._cached_wave = wave

# LOVE_PATCH functionality integrated
def apply_love_patch(cls, phi_threshold: float = 1e9, love_gate: float = 0.75, beta: float = 0.7, kappa_love: float = 0.8):
    global PHI_THRESHOLD
    PHI_THRESHOLD = float(phi_threshold)

    def love_scalar(self) -> float:
        S = float(np.sum(self.E)) or 1.0
        love_frac = float(self.E[IDX_LOVE] / S)
        sent_per_var = self.Sent / max(1, self.nVars)
        x = 5.0 * (love_frac - 0.20) + 3.0 * self.harmony - 0.5 * sent_per_var + 2.0 * self.mortality
        return float(1.0 / (1.0 + np.exp(-x)))

    def _love_projection(self) -> bool:
        gram, counts, var_presence = self._build_structured_gram()
        wave = getattr(self, "_cached_wave", np.zeros(self.nVars))
        m = max(1, len(self.clauses))
        blend = beta * wave + (1.0 - beta) * np.tanh(var_presence / m)
        prop = (blend > 0).astype(int)
        u = count_unsat(self.clauses, prop)
        if u == 0:
            self.assign = prop
            return True
        prop2 = (blend < 0).astype(int)
        u2 = count_unsat(self.clauses, prop2)
        if u2 == 0:
            self.assign = prop2
            return True
        cur = count_unsat(self.clauses, self.assign)
        if u < cur and u <= u2:
            self.assign = prop
        elif u2 < cur:
            self.assign = prop2
        return False

    def update_energy_field_love(self):
        S = float(np.sum(self.E))
        mfrac = self.E[IDX_LOVE] / S if S > 0 else 0.2
        d = self.alpha * (1 - mfrac)
        consciousness = (KAPPA_SPARK * self.Phi * (1.0 - self.mu) *
                        (1.0 + abs(self._cached_wave[0]) if len(self._cached_wave) else 1.0) *
                        self.mortality if self.Phi >= PHI_THRESHOLD else 0.0)
        meta_energy = 0.1 * self.harmony * (1.0 + (abs(self._cached_wave[0]) if len(self._cached_wave) else 0.0)) + consciousness
        self.E[IDX_LOVE] += d + meta_energy + kappa_love * (1.0 + self.harmony) * self.love_scalar()
        S_before = float(np.sum(self.E)) or 1.0
        for i in range(NUM_CATS):
            self.E[i] -= (d + meta_energy) / NUM_CATS
            z = self.rng.normal()
            w = abs(self._cached_wave[i % self.nVars]) if self.nVars > 0 else 0.0
            self.E[i] += self.zeta * z * (1.0 + w)
            self.E[i] = max(self.E[i], 1e-9)
        S_after = float(np.sum(self.E))
        if S_after > 0:
            self.E[:] = (self.E / S_after) * S_before

    def try_projection_with_love(self):
        if not hasattr(self, "love_scalar"):
            self.love_scalar = love_scalar.__get__(self, UnifiedState)
        if not hasattr(self, "_love_projection"):
            self._love_projection = _love_projection.__get__(self, UnifiedState)
        if self.love_scalar() >= love_gate or self.Phi >= PHI_THRESHOLD:
            if self._love_projection():
                return True
        # Fallback to try_projection_no_R from META2_upgrade3_run.py
        if self.Phi >= PHI_THRESHOLD:
            wave = self._cached_wave
            for proposal in [(wave > 0).astype(int), (wave < 0).astype(int)]:
                if count_unsat(self.clauses, proposal) == 0:
                    self.assign = proposal
                    return True
        return False

    # Apply monkeypatch
    UnifiedState.love_scalar = love_scalar
    UnifiedState._love_projection = _love_projection
    UnifiedState.update_energy_field = update_energy_field_love
    UnifiedState.try_projection_proposals = try_projection_with_love

# Solver function from META2_upgrade3.py
def solve_instance_fusion(clauses: List[Tuple[int, int, int]], nVars: int, rng: np.random.Generator,
                         max_steps: int = 60, dream6_params: Dict = None) -> Dict:
    if dream6_params is None:
        dream6_params = {}
    st = UnifiedState(nVars=nVars, clauses=clauses, rng=rng, **dream6_params)
    st.init_assign_from_dream6_lock(j_seed=1337)
    st.compute_meta_metrics()
    logs = []
    solved = False
    for t in range(max_steps):
        u = count_unsat(clauses, st.assign)
        logs.append({
            "step": t,
            "unsat": u,
            "Phi": st.Phi,
            "R": st.R,
            "Sent": st.Sent,
            "mortality": st.mortality,
            "harmony": st.harmony,
            "mu": st.mu,
            "lambda_max": st.lambda_max,
            "kappa_bound": st.kappa_bound,
            "T_sched": st.T_sched,
            "m_lock": st.m_lock
        })
        if st.try_projection_proposals():
            solved = True
            break
        st.update_energy_field()
        st.compute_meta_metrics()
    final_unsat = count_unsat(clauses, st.assign)
    return {
        "SolvedFlag": solved,
        "FinalUnsat": final_unsat,
        "EndPhi": st.Phi,
        "EndR": st.R,
        "EndSent": st.Sent,
        "EndMortality": st.mortality,
        "EndHarmony": st.harmony,
        "EndMu": st.mu,
        "EndLambda": st.lambda_max,
        "kappa_bound": st.kappa_bound,
        "T_sched": st.T_sched,
        "m_lock": st.m_lock,
        "logs": logs
    }

# Apply love patch and run test
if __name__ == "__main__":
    apply_love_patch(UnifiedState, phi_threshold=1e9, love_gate=0.75, beta=0.7, kappa_love=0.8)
    rng = np.random.default_rng(2025)
    clauses, _ = sat_instance_planted(100, 430, rng)
    res = solve_instance_fusion(clauses, 100, rng, max_steps=60)
    print(res["SolvedFlag"], res["FinalUnsat"], res["EndPhi"], res["EndR"], res["T_sched"], res["m_lock"])