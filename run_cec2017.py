"""
run_cec2017.py - CEC 2017 benchmark experiment runner with checkpointing.
Runs in chunks so it can resume across time limits. Saves to results/cec2017.json
"""
import numpy as np, json, os, time, sys
from opfunu.cec_based import cec2017
from psgo import psgo
from competitors import ALL_ALGORITHMS

os.makedirs("results", exist_ok=True)
CKPT = "results/cec2017.json"

# ---- Experiment configuration (tweak these for bigger runs) ----
DIM = 30
FES = 1000 * DIM           # reduced budget; enough to rank algorithms
RUNS = 10                  # paper uses 30; 10 is statistically usable
# 8 representative functions across all 4 categories (uni/multi/hybrid/comp)
FUNC_IDS = [1, 4, 5, 7, 11, 15, 21, 24]

ALGOS = dict(ALL_ALGORITHMS)   # PSO, GWO, WOA, SCA, HHO, AO, AVOA, ARO, INFO, SGA
ALGOS["PSGO"] = psgo
ALGO_NAMES = list(ALGOS.keys())


def load_ckpt():
    if os.path.exists(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {"config": {"DIM": DIM, "FES": FES, "RUNS": RUNS,
                       "FUNC_IDS": FUNC_IDS, "ALGO_NAMES": ALGO_NAMES},
            "results": {}}   # results[f"{fid}"][algo] = [err_run0, err_run1, ...]


def save_ckpt(data):
    with open(CKPT, "w") as f:
        json.dump(data, f, indent=1)


def run_chunk(max_seconds=240):
    data = load_ckpt()
    res = data["results"]
    start = time.time()
    for fid in FUNC_IDS:
        F = getattr(cec2017, f"F{fid}2017")(ndim=DIM)
        fstar = F.f_global
        func = lambda x: F.evaluate(x)
        key = str(fid)
        res.setdefault(key, {})
        for name in ALGO_NAMES:
            res[key].setdefault(name, [])
            while len(res[key][name]) < RUNS:
                seed = len(res[key][name])
                err = ALGOS[name](func, F.lb, F.ub, DIM, max_fes=FES, seed=seed)[1] - fstar
                res[key][name].append(float(max(0.0, err)))
                save_ckpt(data)
                if time.time() - start > max_seconds:
                    print(f"[chunk] time up at F{fid}/{name} "
                          f"({len(res[key][name])}/{RUNS} runs). Saved. Re-run to continue.")
                    return False
        print(f"[chunk] F{fid} complete", flush=True)
    print("[chunk] ALL DONE")
    return True


if __name__ == "__main__":
    secs = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    done = run_chunk(secs)
    sys.exit(0 if done else 2)
