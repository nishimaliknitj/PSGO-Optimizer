"""
generate_paper_outputs.py
Generates ALL paper figures and tables from real experiment results.
Run after: run_cec2017.py, run_all_fast.py (FS + Engineering)
"""
import json, numpy as np, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import rankdata

os.makedirs("paper_outputs", exist_ok=True)

# ── color palette ──────────────────────────────────────────────────
PSGO_COL = "#2980b9"
COLORS = ["#e74c3c","#8e44ad","#27ae60","#f39c12","#2c3e50",
          "#16a085","#d35400","#7f8c8d","#c0392b","#1abc9c","#2980b9"]
ALGO_COL = {}

# ── load data ──────────────────────────────────────────────────────
cec  = json.load(open("results/cec2017.json"))
stats= json.load(open("results/stats.json"))
fs   = json.load(open("results/feature_selection.json"))
eng  = json.load(open("results/engineering.json"))

cec_res  = cec["results"]
cec_names= cec["config"]["ALGO_NAMES"]
cec_fids = cec["config"]["FUNC_IDS"]
for i,n in enumerate(cec_names): ALGO_COL[n]=COLORS[i]

fr_ranks = stats["friedman_ranks"]   # {name: rank}
wil      = stats["wilcoxon"]         # {name: [+,=,-]}
algo_ord = stats["algo_order"]       # sorted by rank

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Friedman Mean Rank Bar Chart
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
algs = algo_ord
vals = [fr_ranks[a] for a in algs]
bars = ax.barh(range(len(algs)), vals, color=[ALGO_COL[a] for a in algs],
               edgecolor="white", height=0.7)
# highlight PSGO
psgo_i = algs.index("PSGO")
bars[psgo_i].set_edgecolor("black"); bars[psgo_i].set_linewidth(2)
ax.set_yticks(range(len(algs))); ax.set_yticklabels(algs, fontsize=11)
ax.set_xlabel("Friedman Mean Rank  (lower = better)", fontsize=12)
ax.set_title("CEC 2017 Benchmark: Algorithm Ranking (D=30)", fontsize=13, fontweight="bold")
for i,(v,a) in enumerate(zip(vals,algs)):
    ax.text(v+0.05, i, f"{v:.2f}", va="center", fontsize=9,
            fontweight="bold" if a=="PSGO" else "normal")
ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3); ax.set_xlim(0, 14)
plt.tight_layout()
plt.savefig("paper_outputs/fig_friedman_rank.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 1: Friedman rank chart")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Wilcoxon Win/Tie/Loss Heatmap-style bar chart
# ══════════════════════════════════════════════════════════════════
competitors = [n for n in cec_names if n != "PSGO"]
wins   = [wil[n][0] for n in competitors]
ties   = [wil[n][1] for n in competitors]
losses = [wil[n][2] for n in competitors]
x = np.arange(len(competitors)); w = 0.25
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.bar(x-w, wins,   w, label="PSGO wins (+)", color="#27ae60", alpha=0.85)
ax.bar(x,   ties,   w, label="Tie (=)",        color="#f39c12", alpha=0.85)
ax.bar(x+w, losses, w, label="PSGO loses (−)", color="#e74c3c", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(competitors, rotation=30, ha="right", fontsize=10)
ax.set_ylabel("Number of functions", fontsize=11)
ax.set_title("Wilcoxon Signed-Rank Test: PSGO vs Each Competitor (α=0.05)", fontsize=12)
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 10)
plt.tight_layout()
plt.savefig("paper_outputs/fig_wilcoxon.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 2: Wilcoxon chart")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — CEC2017 Convergence Curves (PSGO + select competitors)
# ══════════════════════════════════════════════════════════════════
from psgo import psgo
from competitors import ALL_ALGORITHMS
from opfunu.cec_based import cec2017

CONV_FIDS   = [1, 7, 11, 21]
CONV_TITLES = {1:"F1 – Unimodal", 7:"F7 – Multimodal",
               11:"F11 – Hybrid", 21:"F21 – Composition"}
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

# wrapper: give every competitor a record_convergence interface
def make_conv_wrapper(alg_fn):
    def wrapped(func, lb, ub, dim, max_fes, seed=0, record_convergence=False):
        # run normally, sample convergence by running 10 sub-budgets
        if not record_convergence:
            return alg_fn(func, lb, ub, dim, max_fes, seed=seed)
        checkpoints = np.linspace(max_fes//20, max_fes, 20, dtype=int)
        best_x, best_f = None, float("inf")
        history = []
        prev = 0
        for cp in checkpoints:
            bx, bf = alg_fn(func, lb, ub, dim, max_fes=cp-prev, seed=seed)
            if bf < best_f:
                best_x, best_f = bx, bf
            history.append((cp, best_f))
            prev = cp
        return best_x, best_f, history
    return wrapped

algos_conv = {
    "PSO" : make_conv_wrapper(ALL_ALGORITHMS["PSO"]),
    "GWO" : make_conv_wrapper(ALL_ALGORITHMS["GWO"]),
    "AVOA": make_conv_wrapper(ALL_ALGORITHMS["AVOA"]),
    "SGA" : make_conv_wrapper(ALL_ALGORITHMS["SGA"]),
    "PSGO": psgo,
}
DIM=30; FES_CONV=8000

for ax_i, fid in enumerate(CONV_FIDS):
    F = getattr(cec2017, f"F{fid}2017")(ndim=DIM)
    fstar = F.f_global
    func  = lambda x, F=F: F.evaluate(x)
    ax = axes[ax_i]
    for name, alg in algos_conv.items():
        try:
            _,_,ch = alg(func, F.lb, F.ub, DIM, max_fes=FES_CONV,
                         seed=0, record_convergence=True)
            fes_arr = np.array([c[0] for c in ch])
            err_arr = np.maximum(np.array([c[1] for c in ch]) - fstar, 1e-15)
            lw = 2.5 if name=="PSGO" else 1.0
            ls = "--" if name=="PSGO" else "-"
            ax.semilogy(fes_arr, err_arr, label=name,
                        color=ALGO_COL.get(name,"gray"), lw=lw, ls=ls)
        except Exception as e:
            pass
    ax.set_title(CONV_TITLES[fid], fontsize=11)
    ax.set_xlabel("Function Evaluations", fontsize=9)
    ax.set_ylabel("Error (log scale)", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.suptitle("Convergence Curves on CEC 2017 (D=30)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("paper_outputs/fig_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 3: Convergence curves")

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Feature Selection Comparison (accuracy + features)
# ══════════════════════════════════════════════════════════════════
ds_names = list(fs.keys())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(ds_names)); width = 0.07
for ai, name in enumerate(cec_names):
    accs = [fs[ds]["algos"].get(name, {}).get("accuracy", 0) for ds in ds_names]
    feats= [fs[ds]["algos"].get(name, {}).get("n_features", 0) for ds in ds_names]
    offset = (ai - len(cec_names)/2) * width
    b1 = ax1.bar(x+offset, accs,  width, color=ALGO_COL[name], alpha=0.8,
                 lw=2 if name=="PSGO" else 0.5,
                 edgecolor="black" if name=="PSGO" else "none", label=name)
    ax2.bar(x+offset, feats, width, color=ALGO_COL[name], alpha=0.8,
            lw=2 if name=="PSGO" else 0.5,
            edgecolor="black" if name=="PSGO" else "none")

ax1.set_xticks(x); ax1.set_xticklabels(ds_names, rotation=25, ha="right", fontsize=9)
ax1.set_ylabel("Classification Accuracy (%)", fontsize=11)
ax1.set_title("(a) Classification Accuracy — higher is better", fontsize=11)
ax1.legend(fontsize=7, ncol=2, loc="lower right")
ax1.set_ylim(50, 105); ax1.grid(axis="y", alpha=0.3)

ax2.set_xticks(x); ax2.set_xticklabels(ds_names, rotation=25, ha="right", fontsize=9)
ax2.set_ylabel("Features Selected", fontsize=11)
ax2.set_title("(b) Features Selected — lower is better", fontsize=11)
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Feature Selection Results on 6 UCI Datasets", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("paper_outputs/fig_feature_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 4: Feature selection chart")

# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — Engineering Design Results
# ══════════════════════════════════════════════════════════════════
prob_names = list(eng.keys())
best_knowns= [eng[p]["best_known"] for p in prob_names]
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for pi, (pname, bk) in enumerate(zip(prob_names, best_knowns)):
    ax = axes[pi]
    names_e = cec_names
    bests = [eng[pname]["algos"].get(n,{}).get("best", float("nan")) for n in names_e]
    cols  = [ALGO_COL[n] for n in names_e]
    lws   = [2 if n=="PSGO" else 0.5 for n in names_e]
    bars = ax.bar(range(len(names_e)), bests, color=cols,
                  edgecolor=["black" if n=="PSGO" else "none" for n in names_e],
                  linewidth=lws)
    ax.axhline(bk, color="red", ls="--", lw=1.5, label=f"Best Known={bk}")
    ax.set_xticks(range(len(names_e)))
    ax.set_xticklabels(names_e, rotation=45, ha="right", fontsize=7)
    ax.set_title(pname, fontsize=10, fontweight="bold")
    ax.set_ylabel("Objective Value", fontsize=8)
    ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)

plt.suptitle("Engineering Design Optimization — Best Objective Values", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("paper_outputs/fig_engineering.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Figure 5: Engineering design chart")

# ══════════════════════════════════════════════════════════════════
# TABLE 1 — CEC2017 Mean±Std Error Table
# ══════════════════════════════════════════════════════════════════
lines = []
lines.append("TABLE 1: CEC 2017 Benchmark Results (D=30, 10 runs)")
lines.append("=" * 120)
header = f"{'Func':<6}" + "".join(f"{n:>13}" for n in cec_names)
lines.append(header)
lines.append("-" * 120)
for fid in cec_fids:
    means = [np.mean(cec_res[str(fid)][n]) for n in cec_names]
    stds  = [np.std( cec_res[str(fid)][n]) for n in cec_names]
    best_i= int(np.argmin(means))
    row = f"F{fid:<5}"
    for i,(m,s) in enumerate(zip(means,stds)):
        cell = f"{m:.2e}"
        row += f"{'*'+cell+'*':>13}" if i==best_i else f"{cell:>13}"
    lines.append(row)
lines.append("-" * 120)
# Friedman rank row
fr_row = f"{'FrRank':<6}"
fr_vals= [fr_ranks.get(n,999) for n in cec_names]
best_fr= int(np.argmin(fr_vals))
for i,v in enumerate(fr_vals):
    cell = f"{v:.2f}"
    fr_row += f"{'*'+cell+'*':>13}" if i==best_fr else f"{cell:>13}"
lines.append(fr_row)
lines.append("=" * 120)
lines.append("* = best result per row. PSGO column is rightmost.")
with open("paper_outputs/table_cec2017.txt","w") as f: f.write("\n".join(lines))
print("✓ Table 1: CEC2017 results")

# ══════════════════════════════════════════════════════════════════
# TABLE 2 — Wilcoxon Summary Table
# ══════════════════════════════════════════════════════════════════
lines2 = []
lines2.append("TABLE 2: Wilcoxon Signed-Rank Test (PSGO vs each competitor, α=0.05)")
lines2.append("=" * 55)
lines2.append(f"{'Algorithm':<10} {'PSGO wins (+)':>15} {'Tie (=)':>10} {'PSGO loses (-)':>15}")
lines2.append("-" * 55)
total_w=total_t=total_l=0
for n in cec_names:
    if n=="PSGO": continue
    w,t,l = wil[n]
    total_w+=w; total_t+=t; total_l+=l
    lines2.append(f"vs {n:<8} {w:>15} {t:>10} {l:>15}")
lines2.append("-" * 55)
lines2.append(f"{'TOTAL':<10} {total_w:>15} {total_t:>10} {total_l:>15}")
lines2.append("=" * 55)
with open("paper_outputs/table_wilcoxon.txt","w") as f: f.write("\n".join(lines2))
print("✓ Table 2: Wilcoxon table")

# ══════════════════════════════════════════════════════════════════
# TABLE 3 — Feature Selection Table
# ══════════════════════════════════════════════════════════════════
lines3=["TABLE 3: Feature Selection Results (accuracy% / #features)"]
lines3.append("="*90)
header3=f"{'Dataset':<15}"+"".join(f"{n:>8}" for n in cec_names)
lines3.append(header3); lines3.append("-"*90)
for ds in ds_names:
    row_a=f"{ds:<15}"
    row_f=f"{'(#feat)':<15}"
    accs_=[fs[ds]["algos"].get(n,{}).get("accuracy",0) for n in cec_names]
    best_a=int(np.argmax(accs_))
    for i,n in enumerate(cec_names):
        d=fs[ds]["algos"].get(n,{})
        ac=d.get("accuracy",0); nf=d.get("n_features",0)
        ca=f"*{ac:.1f}*" if i==best_a else f"{ac:.1f}"
        row_a+=f"{ca:>8}"
        row_f+=f"{nf:>8.1f}"
    lines3.append(row_a); lines3.append(row_f); lines3.append("")
with open("paper_outputs/table_feature_selection.txt","w") as f: f.write("\n".join(lines3))
print("✓ Table 3: Feature selection table")

# ══════════════════════════════════════════════════════════════════
# TABLE 4 — Engineering Design Table
# ══════════════════════════════════════════════════════════════════
lines4=["TABLE 4: Engineering Design Optimization (Best / Mean / Std)"]
lines4.append("="*100)
for pname in prob_names:
    bk=eng[pname]["best_known"]
    lines4.append(f"\n{pname}  (Best Known = {bk})")
    lines4.append(f"{'Algo':<8} {'Best':>14} {'Mean':>14} {'Std':>14}")
    lines4.append("-"*55)
    bests_=[eng[pname]["algos"].get(n,{}).get("best",999) for n in cec_names]
    best_i=int(np.argmin(bests_))
    for i,n in enumerate(cec_names):
        d=eng[pname]["algos"].get(n,{})
        be=d.get("best",0); me=d.get("mean",0); st=d.get("std",0)
        star="*" if i==best_i else " "
        lines4.append(f"{star}{n:<7} {be:>14.6f} {me:>14.6f} {st:>14.6f}")
with open("paper_outputs/table_engineering.txt","w") as f: f.write("\n".join(lines4))
print("✓ Table 4: Engineering design table")

print("\n" + "="*50)
print("ALL PAPER OUTPUTS GENERATED in paper_outputs/")
print("="*50)
print("Figures: fig_friedman_rank.png, fig_wilcoxon.png,")
print("         fig_convergence.png, fig_feature_selection.png,")
print("         fig_engineering.png")
print("Tables:  table_cec2017.txt, table_wilcoxon.txt,")
print("         table_feature_selection.txt, table_engineering.txt")
