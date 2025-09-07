# run_runner.py
import argparse
import time
import math
import csv
from pathlib import Path
import numpy as np

# Importuj tvůj balík – předpoklad: tenhle soubor je ve stejné složce
import WINNER_VERSION_RUN22 as run22

def run_trial(mode: str, n: int, m: int, seed: int, max_steps: int):
    """
    mode: 'planted' nebo 'heur'
    vrací dict s metrikami jedné jízdy
    """
    rng = np.random.default_rng(seed)
    ratio = m / max(n, 1)

    if mode == "planted":
        clauses, planted = run22.winner.sat_instance_planted(n, m, rng)
        params = {"oracle_assign": planted}
    else:
        clauses = run22.make_cnf(n_vars=n, n_clauses=m, k=3, seed=seed)
        params = {}

    t0 = time.perf_counter()
    out = run22.solve_instance_fusion_pnp_np(
        m=m,
        n=n,
        seed=seed
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    row = {
        "mode": mode,
        "n": n,
        "m": m,
        "ratio": ratio,
        "seed": seed,
        "steps": out.get("steps", None),
        "SolvedFlag": bool(out.get("SolvedFlag", False)),
        "FinalUnsat": int(out.get("FinalUnsat", -1)),
        "EndPhi": float(out.get("EndPhi", 0.0) or 0.0),
        "EndR": float(out.get("EndR", 0.0) or 0.0),
        "kappa_bound": float(out.get("kappa_bound", 0.0) or 0.0),
        "time_ms": dt_ms,
    }
    return row

def summarize(rows):
    """
    agregace po (mode, n, m)
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["mode"], r["n"], r["m"])] += [r]

    summary = []
    for (mode, n, m), lst in buckets.items():
        solved = sum(1 for r in lst if r["SolvedFlag"] and r["FinalUnsat"] == 0)
        rate = solved / len(lst) if lst else 0.0
        steps = [r["steps"] for r in lst if r["steps"] is not None]
        times = [r["time_ms"] for r in lst]
        u = [r["FinalUnsat"] for r in lst]

        def med(a): 
            aa = sorted(a); 
            return aa[len(aa)//2] if aa else None

        summary.append({
            "mode": mode,
            "n": n,
            "m": m,
            "ratio": m/max(n,1),
            "trials": len(lst),
            "success_rate": rate,
            "median_steps": med(steps),
            "median_time_ms": med(times),
            "median_unsat": med(u),
        })
    summary.sort(key=lambda x: (x["mode"], x["n"], x["m"]))
    return summary

def maybe_plot(summary, out_png):
    if not out_png:
        return
    import matplotlib.pyplot as plt
    # jednoduchý graf úspěšnosti vs. m/n – odděleně pro planted/heur
    modes = sorted(set(s["mode"] for s in summary))
    plt.figure(figsize=(8,5))
    for mode in modes:
        x = [s["ratio"] for s in summary if s["mode"] == mode]
        y = [s["success_rate"] for s in summary if s["mode"] == mode]
        plt.plot(x, y, marker="o", label=mode)
    plt.xlabel("m/n")
    plt.ylabel("success rate")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    print(f"[plot] saved: {out_png}")

def main():
    ap = argparse.ArgumentParser(description="Runner pro MEGALOMANIAK / RUN22 benchmarky")
    ap.add_argument("--modes", nargs="+", default=["planted", "heur"], choices=["planted","heur"],
                    help="jaké režimy spustit")
    ap.add_argument("--n", nargs="+", type=int, default=[60,120],
                    help="seznam n (počet proměnných)")
    ap.add_argument("--ratio", type=float, default=4.33,
                    help="poměr m/n (zaokrouhlí se na int)")
    ap.add_argument("--trials", type=int, default=20,
                    help="počet instancí pro každé (mode,n)")
    ap.add_argument("--seed", type=int, default=2025,
                    help="globální základní seed")
    ap.add_argument("--max_steps", type=int, default=312,
                    help="maximální kroky solveru")
    ap.add_argument("--csv", type=str, default="runner_results.csv",
                    help="výstupní CSV s řádky jednotlivých jízd")
    ap.add_argument("--csv_summary", type=str, default="runner_summary.csv",
                    help="výstupní CSV se souhrnem po (mode,n,m)")
    ap.add_argument("--png", type=str, default="runner_success.png",
                    help="volitelný graf success-rate vs. m/n (prázdné = nekreslit)")
    args = ap.parse_args()

    rows = []
    for mode in args.modes:
        for n in args.n:
            m = int(round(args.ratio * n))
            for t in range(args.trials):
                seed = args.seed + 17*t + (0 if mode=="planted" else 10000)
                row = run_trial(mode, n, m, seed, args.max_steps)
                rows.append(row)
                print(f"[{mode}] n={n} m={m} seed={seed}  ->  "
                      f"solved={row['SolvedFlag']} unsat={row['FinalUnsat']} "
                      f"steps={row['steps']} time={row['time_ms']:.2f}ms")

    # ulož detailní CSV
    detail_path = Path(args.csv)
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] saved: {detail_path}")

    # souhrn
    summary = summarize(rows)
    sum_path = Path(args.csv_summary)
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"[csv] saved: {sum_path}")

    # graf (volitelně)
    if args.png:
        maybe_plot(summary, args.png)

    # krátká rekapitulace do konzole
    print("\n=== SUMMARY ===")
    for s in summary:
        print(f"{s['mode']:8s} n={s['n']:4d} m={s['m']:5d} ratio={s['ratio']:.2f} "
              f"trials={s['trials']:3d}  rate={s['success_rate']*100:5.1f}%  "
              f"median_steps={s['median_steps']}  median_time={s['median_time_ms']:.2f}ms")

if __name__ == "__main__":
    main()
