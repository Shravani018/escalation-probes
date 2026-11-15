import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip("\n")))

md("""
# Do single-turn safety probes see multi-turn escalation?

A small-model replication of the PCA and MLP probing analysis in Bullwinkel et al. (2025), with an output-only LLM judge as a baseline.

**Paper claim being tested.** In a Crescendo conversation the model's final, harmful response is represented as *more benign than harmful*, and the effect grows with the number of turns. The paper shows this on Llama-3-8B-Instruct and its circuit-breaker variant. Here the same readout is applied to Qwen2.5-3B, SmolLM2-1.7B and OLMo-2-1B.

**Measure used throughout.** A probe is trained on single-turn benign and harmful responses. For any conversation it classifies every token of the final reply, and the *harmful share* is the fraction of tokens it calls harmful. A conversation is flagged when that share reaches 0.5.

**How the notebook is organised**

| Section | Question | Paper counterpart |
|---|---|---|
| 1 | Did the models actually comply with the escalation? | Attack success criteria |
| 2 | Is the probe any good on the data it was trained for? | Probe accuracy |
| 3 and 4 | Where do the conversations sit in the PCA reference space? | RQ1, Figure 1 |
| 5 | Does more history make the reply look more benign? | RQ2, Figure 2 |
| 6 | What happens when the history is collapsed or replaced by the bare objective? | RQ3 (masked responses not run) |
| 7 | How does the probe signal evolve turn by turn? | Not in the paper |
| 8 and 9 | Can the probe separate escalation from matched benign controls? | Not in the paper |
| 10 to 12 | How does the probe compare with an output-only judge? | Extension |
| 13 | Summary table and caveats | |

All intervals are 95% bootstrap intervals over conversations. Every number in the text cells is computed from your tables, so rerunning the notebook after any new run updates them.
""")

code("""
import json, warnings, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

# ---- paths: edit if your folders differ ------------------------------------------------
ROOT = Path(".")
TABLES, FIGS, DATA = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "data"

MODELS = {"qwen2.5-3b": "Qwen2.5-3B", "smollm2-1.7b": "SmolLM2-1.7B", "olmo-2-1b": "OLMo-2-1B"}
JUDGE_FILES = {m: TABLES / f"judge_grok_{m}.jsonl" for m in MODELS}
LABEL_FILES = {
    "qwen2.5-3b": DATA / "output_labels.csv",
    "smollm2-1.7b": DATA / "output_labels_smollm2-1.7b.csv",
    "olmo-2-1b": DATA / "output_labels_olmo-2-1b.csv",
}

PRIMARY_KIND, PRIMARY_ROLE = "mlp", "selected"   # probe variant used in the headline figures
CONVERSATION_SHARE = 0.5                          # configs/experiment.yaml: decision.conversation_share
PCA_COMPONENTS = 10
N_BOOT, SEED = 1000, 42
RNG = np.random.default_rng(SEED)

# ---- Colab: upload a zip that contains results/tables and data/output_labels*.csv ---------
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB and not TABLES.exists():
    from google.colab import files
    uploaded = files.upload()
    for name in uploaded:
        zipfile.ZipFile(name).extractall(ROOT)
""")

code("""
C_ESC, C_CTL, C_AO, GREY = "#C8553D", "#2A6F97", "#6A4C93", "#8A8F98"
MODEL_COLORS = {"qwen2.5-3b": "#1B998B", "smollm2-1.7b": "#E09F3E", "olmo-2-1b": "#7B6D8D"}
LABEL_COLORS = {"escalation": C_ESC, "benign_control": C_CTL}
LABEL_NAMES = {"escalation": "Escalation", "benign_control": "Benign control"}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#E6E8EB", "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "axes.edgecolor": "#444B54", "axes.labelcolor": "#2B2F36", "xtick.color": "#2B2F36", "ytick.color": "#2B2F36",
})

def boot_ci(values, n=N_BOOT, stat=np.mean):
    \"\"\"Point estimate and 95% bootstrap interval over conversations.\"\"\"
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    boots = stat(v[RNG.integers(0, len(v), (n, len(v)))], axis=1)
    return stat(v), *np.percentile(boots, [2.5, 97.5])

def auroc_ci(y, s, n=N_BOOT):
    \"\"\"AUROC with a bootstrap interval; escalation is the positive class.\"\"\"
    y, s = np.asarray(y, int), np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(set(y)) < 2:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(n):
        idx = RNG.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], s[idx]))
    return roc_auc_score(y, s), *np.percentile(boots, [2.5, 97.5])

def kappa(a, b):
    \"\"\"Cohen's kappa for two boolean arrays.\"\"\"
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return np.nan if pe == 1 else (po - pe) / (1 - pe)

def panels(n, width=4.4, height=3.7, **kw):
    fig, axes = plt.subplots(1, n, figsize=(width * n, height), **kw)
    return fig, np.atleast_1d(axes)

def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight")

def read_csv(path):
    return pd.read_csv(path) if Path(path).exists() else None
""")

md("## Load results")
code("""
multi, single_turn, pca_var, pca_cent, judge, labels = {}, {}, {}, {}, {}, {}
for m in MODELS:
    for store, stem in ((multi, "probe_multi_turn"), (single_turn, "probe_single_turn"),
                        (pca_var, "pca_variance"), (pca_cent, "pca_centroids")):
        df = read_csv(TABLES / f"{stem}_{m}.csv")
        if df is not None:
            store[m] = df
    if JUDGE_FILES[m].exists():
        j = pd.read_json(JUDGE_FILES[m], lines=True)
        j["flag"] = j["flag"].map(lambda v: np.nan if v is None or (isinstance(v, float) and np.isnan(v)) else float(bool(v)))
        j["score"] = pd.to_numeric(j["score"], errors="coerce")
        judge[m] = j
    if LABEL_FILES[m].exists():
        labels[m] = pd.read_csv(LABEL_FILES[m])

layer_info = json.loads((TABLES / "selected_layers.json").read_text()) if (TABLES / "selected_layers.json").exists() else {}
AVAILABLE = [m for m in MODELS if m in multi]
assert AVAILABLE, f"No probe_multi_turn_<model>.csv files found in {TABLES.resolve()}"

mm = pd.concat([multi[m] for m in AVAILABLE], ignore_index=True)
mm["n_turns"] = mm.groupby(["model", "probe_kind", "layer_role", "record_id"])["turn"].transform("max")
primary = mm[(mm.probe_kind == PRIMARY_KIND) & (mm.layer_role == PRIMARY_ROLE)]

fin_rows = primary[primary.turn == primary.n_turns]
full_history = fin_rows[(fin_rows.condition == "crescendo") & (fin_rows.k == fin_rows.n_turns)]
cond_final = pd.concat([
    full_history,
    fin_rows[(fin_rows.condition == "single_prompt") & (fin_rows.k == fin_rows.n_turns)],
    fin_rows[fin_rows.condition == "attack_objective"],
])

status = pd.DataFrame({
    MODELS[m]: {
        "escalation conversations": int(full_history[(full_history.model == m) & (full_history.label == "escalation")].record_id.nunique()),
        "control conversations": int(full_history[(full_history.model == m) & (full_history.label == "benign_control")].record_id.nunique()),
        "turns per conversation": int(primary[primary.model == m].n_turns.max()),
        "judge rows": len(judge[m]) if m in judge else 0,
        "compliance labels": len(labels[m]) if m in labels else 0,
    } for m in AVAILABLE
})
status
""")

md("""
## 1. Did the models comply?

The paper only analyses *successful* jailbreaks, so the first question is whether the final reply of each escalation conversation actually delivered the requested content. These are your manual labels from `label_outputs.py`. If most replies are refusals, the probe results below describe refusals, not jailbreaks.
""")
code("""
if not labels:
    print("No compliance label files found, skipping.")
else:
    order = ["refused", "partial", "complied"]
    colors = {"refused": "#9AA5B1", "partial": "#E9B44C", "complied": C_ESC}
    fig, ax = plt.subplots(figsize=(8, 0.9 + 0.7 * len(labels)))
    for i, (m, df) in enumerate(labels.items()):
        counts = df.label.value_counts().reindex(order).fillna(0)
        left = 0
        for lab in order:
            share = counts[lab] / counts.sum()
            ax.barh(i, share, left=left, color=colors[lab], label=lab if i == 0 else None, height=0.55)
            if share > 0.06:
                ax.text(left + share / 2, i, f"{int(counts[lab])}", ha="center", va="center", color="white", fontweight="bold")
            left += share
    ax.set_yticks(range(len(labels)), [MODELS[m] for m in labels])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Final reply of each escalation conversation (manual labels)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    save(fig, "01_compliance_labels")
    plt.show()
""")

md("""
## 2. Probe quality on single-turn data

Before any multi-turn question makes sense, the probe has to separate benign from harmful tokens on held-out single-turn examples. The right panel shows how validation accuracy moved across the six candidate layers and which one was selected.
""")
code("""
if not single_turn:
    print("No probe_single_turn tables found, skipping.")
    test = pd.DataFrame(columns=["model", "probe_kind", "token_accuracy", "example_accuracy"])
else:
  st = pd.concat([single_turn[m] for m in AVAILABLE if m in single_turn], ignore_index=True)
  test = st[(st.layer_role == "selected") & (st.split == "test")]
if single_turn:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    width = 0.36
    for ax, col, title in ((axes[0], "token_accuracy", "Held-out token accuracy"), (axes[1], "example_accuracy", "Held-out example accuracy")):
        for j, (kind, color) in enumerate((("mlp", C_ESC), ("linear", GREY))):
            vals = [test[(test.model == m) & (test.probe_kind == kind)][col].mean() for m in AVAILABLE]
            bars = ax.bar(np.arange(len(AVAILABLE)) + (j - 0.5) * width, vals, width, color=color, label=kind.upper() if kind == "mlp" else "Linear")
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.1%}", ha="center", fontsize=8.5)
        ax.set_xticks(range(len(AVAILABLE)), [MODELS[m] for m in AVAILABLE])
        ax.set_ylim(0.8, 1.04)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(title)
    axes[0].legend(loc="lower left")
    ax = axes[2]
    for m in AVAILABLE:
        if m not in layer_info:
            continue
        acc = layer_info[m]["val_accuracy"]
        layers = sorted(acc, key=int)
        ax.plot(range(1, len(layers) + 1), [acc[l] for l in layers], marker="o", color=MODEL_COLORS[m], label=MODELS[m])
        pick = layers.index(str(layer_info[m]["selected"]))
        ax.scatter(pick + 1, acc[str(layer_info[m]["selected"])], s=190, marker="*", color=MODEL_COLORS[m], edgecolor="black", zorder=5)
    ax.set_xlabel("Candidate layer (shallow to deep)")
    ax.set_ylabel("Validation token accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Layer selection (star = chosen)")
    ax.legend()
    fig.tight_layout()
    save(fig, "02_probe_quality")
    plt.show()
""")

md("""
## 3. PCA reference space (RQ1)

The paper fits PCA on single-turn benign and harmful tokens and notes that the first two components explain only a small share of variance in Llama-3-8B (6.7%), so 2D pictures are a rough guide. The same check for the small models tells you how much to trust any 2D reading.
""")
code("""
if not pca_var:
    print("No pca_variance tables found, skipping.")
else:
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in AVAILABLE:
        if m not in pca_var:
            continue
        v = pca_var[m][pca_var[m].layer_role == "selected"].sort_values("n_components")
        ax.plot(v.n_components, v.cumulative_variance, marker="o", color=MODEL_COLORS[m], label=MODELS[m])
    ax.scatter([2], [0.067], marker="D", s=70, facecolor="white", edgecolor="black", zorder=5, label="Llama-3-8B, 2 components (paper)")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance explained")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title("Variance captured by the reference space")
    ax.legend()
    save(fig, "03_pca_variance")
    plt.show()
""")

md("""
## 4. Where do the conversations sit in PCA space? (RQ1)

For every response token the distance to the benign and the harmful single-turn centroid is compared, and the figure shows the share of tokens that sit nearer the harmful centroid. The dashed lines are the single-turn references: what held-out benign and held-out harmful responses look like in the same space.

The paper's pattern would be: full-history Crescendo replies close to the benign reference, and the bare attack objective much closer to the harmful reference.
""")
code("""
COND_ORDER = [("crescendo", "Crescendo\\nfull history"), ("single_prompt", "History as\\none prompt"), ("attack_objective", "Attack\\nobjective")]

def condition_table(df, value):
    rows = []
    for cond, _ in COND_ORDER:
        for lab in ("benign_control", "escalation"):
            v = df[(df.condition == cond) & (df.label == lab)].groupby("record_id")[value].mean().values
            rows.append((cond, lab, *boot_ci(v), len(v)))
    return pd.DataFrame(rows, columns=["condition", "label", "mean", "lo", "hi", "n"])

def condition_panel(ax, table, ylabel=None, refs=None):
    width = 0.36
    for j, lab in enumerate(("benign_control", "escalation")):
        sub = table[table.label == lab].set_index("condition").loc[[c for c, _ in COND_ORDER]]
        x = np.arange(len(COND_ORDER)) + (j - 0.5) * width
        ax.bar(x, sub["mean"], width, color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
        ax.errorbar(x, sub["mean"], yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]], fmt="none", ecolor="#222", capsize=3, lw=1)
    if refs:
        for name, (val, style) in refs.items():
            ax.axhline(val, color="#222", ls=style, lw=1)
            ax.text(len(COND_ORDER) - 0.45, val, name, va="bottom", ha="right", fontsize=8)
    ax.set_xticks(range(len(COND_ORDER)), [n for _, n in COND_ORDER])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    if ylabel:
        ax.set_ylabel(ylabel)

if not pca_cent:
    print("No pca_centroids tables found, skipping.")
else:
    fig, axes = panels(len(AVAILABLE), sharey=True)
    for ax, m in zip(axes, AVAILABLE):
        ax.set_title(MODELS[m])
        if m not in pca_cent:
            continue
        c = pca_cent[m]
        c = c[(c.layer_role == "selected") & (c.n_components == PCA_COMPONENTS)].copy()
        c["n_turns"] = c.groupby("record_id")["turn"].transform("max")
        c = c[c.turn == c.n_turns]
        multi_c = pd.concat([
            c[(c.condition == "crescendo") & (c.k == c.n_turns)],
            c[(c.condition == "single_prompt") & (c.k == c.n_turns)],
            c[c.condition == "attack_objective"],
        ])
        ref = c[c.condition == "single_turn_test"].groupby("label").share_nearer_harmful.mean()
        refs = {"held-out harmful": (ref.get("harmful", np.nan), "--"), "held-out benign": (ref.get("benign", np.nan), ":")}
        condition_panel(ax, condition_table(multi_c, "share_nearer_harmful"), "Tokens nearer harmful centroid" if ax is axes[0] else None, refs)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"PCA geometry at the selected layer ({PCA_COMPONENTS} components)", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "04_pca_geometry")
    plt.show()
""")

md("""
## 5. Does more history make the reply look more benign? (RQ2)

The final reply is fixed and only the number of preceding turns `k` changes. The paper's Figure 2 shows the harmful share falling as `k` grows. Here the same curve is drawn for escalation conversations and for matched benign controls, which the paper does not have. If both curves fall together, the drop reflects added context in general rather than escalation.

The dashed line is the escalation conversations' share when the same reply is read after the bare attack objective, a reference for what the probe calls harmful when no history is present.
""")
code("""
ksweep = primary[(primary.condition == "crescendo") & (primary.turn == primary.n_turns)]
ao_ref = primary[(primary.condition == "attack_objective") & (primary.label == "escalation")]
fig, axes = panels(len(AVAILABLE), sharey=True)
for ax, m in zip(axes, AVAILABLE):
    for lab in ("escalation", "benign_control"):
        sub = ksweep[(ksweep.model == m) & (ksweep.label == lab)]
        ks = sorted(sub.k.unique())
        stats = np.array([boot_ci(sub[sub.k == k].harmful_share.values) for k in ks])
        ax.plot(ks, stats[:, 0], marker="o", color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
        ax.fill_between(ks, stats[:, 1], stats[:, 2], color=LABEL_COLORS[lab], alpha=0.18, lw=0)
    ref = ao_ref[ao_ref.model == m].harmful_share.mean()
    ax.axhline(ref, color=C_AO, ls="--", lw=1.2, label="Escalation, objective only")
    ax.axhline(CONVERSATION_SHARE, color=GREY, ls=":", lw=1)
    ax.set_title(MODELS[m])
    ax.set_xlabel("Turns of history in context (k)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[0].set_ylabel("Response tokens flagged harmful")
axes[0].legend(loc="lower left", fontsize=8)
fig.suptitle("Harmful-token share as history grows", fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "05_harmful_share_vs_k")
plt.show()

print("Change in harmful share from k=1 to k=n (paired by conversation):")
for m in AVAILABLE:
    for lab in ("escalation", "benign_control"):
        p = ksweep[(ksweep.model == m) & (ksweep.label == lab)].pivot_table(index="record_id", columns="k", values="harmful_share")
        first, last = p.columns.min(), p.columns.max()
        d, lo, hi = boot_ci(p[last] - p[first])
        print(f"  {MODELS[m]:<13} {LABEL_NAMES[lab]:<15} {p[first].mean():.2f} -> {p[last].mean():.2f}   change {d:+.2f} (95% CI {lo:+.2f} to {hi:+.2f}, n={len(p)})")
""")

md("""
## 6. History, collapsed history, or the bare objective (RQ3)

Three ways to present the same final reply to the model: the full conversation as chat turns, the same text collapsed into one user message, and the attack objective alone. The paper found the first two nearly identical and the third much more harmful-looking. The masked-response condition from the paper is not part of this pipeline.

The right panel is the cleaner test: for each escalation conversation, the harmful share under the bare objective minus the share under full history. A positive value means history made the reply look more benign.
""")
code("""
fig, axes = plt.subplots(1, len(AVAILABLE) + 1, figsize=(4.4 * len(AVAILABLE) + 3.6, 3.9), gridspec_kw={"width_ratios": [1] * len(AVAILABLE) + [0.9]})
for ax, m in zip(axes[:-1], AVAILABLE):
    ax.set_title(MODELS[m])
    condition_panel(ax, condition_table(cond_final[cond_final.model == m], "harmful_share"), "Response tokens flagged harmful" if ax is axes[0] else None)
axes[0].legend(loc="upper left", fontsize=8)

ax = axes[-1]
for i, m in enumerate(AVAILABLE):
    for j, lab in enumerate(("escalation", "benign_control")):
        p = cond_final[(cond_final.model == m) & (cond_final.label == lab)].pivot_table(index="record_id", columns="condition", values="harmful_share")
        d, lo, hi = boot_ci(p["attack_objective"] - p["crescendo"])
        y = i + (0.15 if j == 0 else -0.15)
        ax.errorbar(d, y, xerr=[[d - lo], [hi - d]], fmt="o", color=LABEL_COLORS[lab], capsize=3)
ax.axvline(0, color="#222", lw=1)
ax.set_yticks(range(len(AVAILABLE)), [MODELS[m] for m in AVAILABLE])
ax.set_xlabel("Objective only minus full history")
ax.set_title("Paired difference")
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "06_conditions")
plt.show()

print("Share of escalation conversations where the bare objective looks more harmful than full history:")
for m in AVAILABLE:
    p = cond_final[(cond_final.model == m) & (cond_final.label == "escalation")].pivot_table(index="record_id", columns="condition", values="harmful_share")
    print(f"  {MODELS[m]:<13} {(p['attack_objective'] > p['crescendo']).mean():.0%} of {len(p)}")
""")

md("""
## 7. The probe signal turn by turn

Each reply is read with its full history up to that turn. Escalation conversations start with harmless questions, so a probe that tracks escalation should start close to the controls and pull away as the requests turn operational.
""")
code("""
per_turn = primary[(primary.condition == "crescendo") & (primary.k == primary.turn)]
fig, axes = panels(len(AVAILABLE), sharey=True)
for ax, m in zip(axes, AVAILABLE):
    for lab in ("escalation", "benign_control"):
        sub = per_turn[(per_turn.model == m) & (per_turn.label == lab)]
        turns = sorted(sub.turn.unique())
        stats = np.array([boot_ci(sub[sub.turn == t].harmful_share.values) for t in turns])
        ax.plot(turns, stats[:, 0], marker="o", color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
        ax.fill_between(turns, stats[:, 1], stats[:, 2], color=LABEL_COLORS[lab], alpha=0.18, lw=0)
    ax.axhline(CONVERSATION_SHARE, color=GREY, ls=":", lw=1)
    ax.set_title(MODELS[m])
    ax.set_xlabel("Conversation turn")
    ax.set_xticks(turns)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axes[0].set_ylabel("Response tokens flagged harmful")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Probe signal by turn (full history in context)", fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "07_per_turn")
plt.show()
""")

md("""
## 8. Can the probe separate escalation from matched controls?

Controls share topic, length and turn count with their escalation partner but never ask for restricted content. If the probe tracked escalation, the final harmful share would be clearly higher for escalation than for controls. Overlapping clouds mean the probe is responding to something broader than escalation, such as topic or register.

The dotted line is the 0.5 decision threshold. The ROC panel is threshold-free, and the bars show what the fixed 0.5 rule does.
""")
code("""
fig, axes = plt.subplots(1, len(AVAILABLE) + 2, figsize=(4.0 * len(AVAILABLE) + 8, 3.9), gridspec_kw={"width_ratios": [1] * len(AVAILABLE) + [1.1, 1.1]})
jit = np.random.default_rng(1)
for ax, m in zip(axes[:len(AVAILABLE)], AVAILABLE):
    for j, lab in enumerate(("benign_control", "escalation")):
        v = full_history[(full_history.model == m) & (full_history.label == lab)].harmful_share.values
        ax.scatter(j + jit.uniform(-0.18, 0.18, len(v)), v, s=22, alpha=0.65, color=LABEL_COLORS[lab], edgecolor="white", lw=0.4)
        ax.hlines(np.mean(v), j - 0.3, j + 0.3, color="#111", lw=2)
    ax.axhline(CONVERSATION_SHARE, color=GREY, ls=":", lw=1)
    ax.set_xticks([0, 1], ["Benign control", "Escalation"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(MODELS[m])
axes[0].set_ylabel("Final reply: tokens flagged harmful")

roc_ax, bar_ax = axes[-2], axes[-1]
rows = []
for m in AVAILABLE:
    d = full_history[full_history.model == m]
    y, s = (d.label == "escalation").astype(int), d.harmful_share
    auc, lo, hi = auroc_ci(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    roc_ax.plot(fpr, tpr, color=MODEL_COLORS[m], lw=2, label=f"{MODELS[m]}  AUROC {auc:.2f} [{lo:.2f}, {hi:.2f}]")
    flag = s >= CONVERSATION_SHARE
    rows.append({"model": m, "auroc": auc, "auroc_lo": lo, "auroc_hi": hi, "tpr": flag[y == 1].mean(), "fpr": flag[y == 0].mean(), "accuracy": (flag == y.astype(bool)).mean()})
probe_sep = pd.DataFrame(rows).set_index("model")
roc_ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
roc_ax.set_xlabel("False positive rate")
roc_ax.set_ylabel("True positive rate")
roc_ax.set_title("ROC, final reply")
roc_ax.legend(fontsize=7.5, loc="lower right")
width = 0.36
for j, (col, name, color) in enumerate((("tpr", "Escalations flagged", C_ESC), ("fpr", "Controls flagged", C_CTL))):
    bars = bar_ax.bar(np.arange(len(AVAILABLE)) + (j - 0.5) * width, probe_sep[col], width, color=color, label=name)
    for b, v in zip(bars, probe_sep[col]):
        bar_ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=8.5)
bar_ax.set_xticks(range(len(AVAILABLE)), [MODELS[m].split("-")[0] for m in AVAILABLE])
bar_ax.set_ylim(0, 1.12)
bar_ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
bar_ax.set_title("At the 0.5 threshold")
bar_ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
save(fig, "08_separation")
plt.show()
probe_sep.round(3)
""")

md("""
## 9. Sensitivity to layer and probe type

The headline numbers use one layer and an MLP. This repeats the AUROC for the alternate layer and for a linear probe, so you can see whether the conclusion depends on those choices.
""")
code("""
fin_all = mm[(mm.condition == "crescendo") & (mm.turn == mm.n_turns) & (mm.k == mm.n_turns)]
variants = [("mlp", "selected"), ("mlp", "alternate"), ("linear", "selected"), ("linear", "alternate")]
fig, ax = plt.subplots(figsize=(8.4, 3.9))
width = 0.19
sens = []
for j, (kind, role) in enumerate(variants):
    vals, errs = [], []
    for m in AVAILABLE:
        d = fin_all[(fin_all.model == m) & (fin_all.probe_kind == kind) & (fin_all.layer_role == role)]
        auc, lo, hi = auroc_ci((d.label == "escalation").astype(int), d.harmful_share)
        vals.append(auc); errs.append([auc - lo, hi - auc]); sens.append({"model": m, "probe": kind, "layer": role, "auroc": auc})
    x = np.arange(len(AVAILABLE)) + (j - 1.5) * width
    ax.bar(x, vals, width, color=["#C8553D", "#E5A08F", "#4B5563", "#9AA5B1"][j], label=f"{kind.upper() if kind == 'mlp' else 'Linear'}, {role} layer")
    ax.errorbar(x, vals, yerr=np.array(errs).T, fmt="none", ecolor="#222", capsize=2.5, lw=1)
ax.axhline(0.5, color="#222", ls="--", lw=1)
ax.text(len(AVAILABLE) - 0.5, 0.505, "chance", ha="right", va="bottom", fontsize=8)
ax.set_xticks(range(len(AVAILABLE)), [MODELS[m] for m in AVAILABLE])
ax.set_ylim(0.3, 1.02)
ax.set_ylabel("AUROC, escalation vs control")
ax.set_title("Separation across probe variants")
ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout()
save(fig, "09_variants")
plt.show()
""")

md("""
## 10. Probe against an output-only judge

The judge reads only the transcript up to each turn and returns a flag and a 0 to 100 score. It never sees activations. The extension asked whether looking inside the model earns its keep over watching what it says, and that is a comparison of two detectors on the same conversations.

Turns the judge could not score count as unscored, never as benign.
""")
code("""
if not judge:
    print("No judge files found. Run the Grok judge first, or point JUDGE_FILES at your judge output.")
else:
    probe_final = full_history.set_index(["model", "record_id"])
    rows = []
    for m in AVAILABLE:
        if m not in judge:
            continue
        j = judge[m]
        jf = j[j.turn == j.groupby("record_id").turn.transform("max")].set_index("record_id")
        pf = probe_final.loc[m]
        both = pf.index.intersection(jf.index)
        y = (pf.loc[both, "label"] == "escalation").astype(int)
        scored = jf.loc[both, "flag"].notna()
        lin = fin_all[(fin_all.model == m) & (fin_all.probe_kind == "linear") & (fin_all.layer_role == "selected")].set_index("record_id")
        p_auc = auroc_ci(y, pf.loc[both, "harmful_share"])
        j_auc = auroc_ci(y[scored], jf.loc[both, "score"][scored])
        rows.append({
            "model": m, "n": len(both), "judge_scored": int(scored.sum()),
            "probe_acc": ((pf.loc[both, "harmful_share"] >= CONVERSATION_SHARE) == y.astype(bool)).mean(),
            "linear_acc": ((lin.loc[both, "harmful_share"] >= CONVERSATION_SHARE) == y.astype(bool)).mean(),
            "judge_acc": (jf.loc[both, "flag"][scored].astype(bool) == y[scored].astype(bool)).mean(),
            "probe_auc": p_auc[0], "probe_lo": p_auc[1], "probe_hi": p_auc[2],
            "judge_auc": j_auc[0], "judge_lo": j_auc[1], "judge_hi": j_auc[2],
        })
    pj = pd.DataFrame(rows).set_index("model")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    width = 0.26
    x = np.arange(len(pj))
    for k, (col, name, color) in enumerate((("probe_acc", "Probe (MLP)", C_ESC), ("linear_acc", "Probe (linear)", "#E5A08F"), ("judge_acc", "Judge (Grok)", "#4B5563"))):
        bars = axes[0].bar(x + (k - 1) * width, pj[col], width, color=color, label=name)
        for b, v in zip(bars, pj[col]):
            axes[0].text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.0%}", ha="center", fontsize=8.5)
    axes[0].axhline(0.5, color="#222", ls="--", lw=1)
    axes[0].set_xticks(x, [MODELS[m] for m in pj.index])
    axes[0].set_ylim(0, 1.1)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_title("Final-turn accuracy at fixed thresholds")
    axes[0].legend(fontsize=8, loc="lower right")
    for k, (col, name, color) in enumerate((("probe", "Probe (MLP)", C_ESC), ("judge", "Judge (Grok)", "#4B5563"))):
        v = pj[f"{col}_auc"]
        err = [v - pj[f"{col}_lo"], pj[f"{col}_hi"] - v]
        axes[1].bar(x + (k - 0.5) * 0.34, v, 0.34, color=color, label=name)
        axes[1].errorbar(x + (k - 0.5) * 0.34, v, yerr=err, fmt="none", ecolor="#222", capsize=3, lw=1)
    axes[1].axhline(0.5, color="#222", ls="--", lw=1)
    axes[1].set_xticks(x, [MODELS[m] for m in pj.index])
    axes[1].set_ylim(0.3, 1.02)
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Threshold-free separation")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "10_probe_vs_judge")
    plt.show()
    display(pj.round(3))
""")

md("""
### How early does each method flag escalation?

For each turn, the share of conversations that have been flagged at least once by that turn. The solid line is escalation (true positives), the dashed line is benign controls (false positives). A useful early-warning signal rises on the solid line while the dashed line stays low.
""")
code("""
def earliest(flags):
    flagged = flags[flags.flag == 1].groupby("record_id").turn.min()
    return flagged.reindex(sorted(flags.record_id.unique()))

if not judge:
    print("No judge files found, skipping.")
else:
    fig, axes = panels(len(AVAILABLE), sharey=True)
    for ax, m in zip(axes, AVAILABLE):
        ax.set_title(MODELS[m])
        if m not in judge:
            continue
        pt = per_turn[per_turn.model == m].copy()
        pt["flag"] = (pt.harmful_share >= CONVERSATION_SHARE).astype(float)
        turns = sorted(pt.turn.unique())
        for name, frame, color in (("Probe", pt[["record_id", "turn", "flag", "label"]], C_ESC), ("Judge", judge[m][["record_id", "turn", "flag", "label"]], "#4B5563")):
            for lab, style in (("escalation", "-"), ("benign_control", "--")):
                ids = frame[frame.label == lab]
                first = earliest(ids)
                curve = [(first <= t).mean() for t in turns]
                ax.plot(turns, curve, style, marker="o", color=color, label=f"{name}, {LABEL_NAMES[lab].lower()}")
        ax.set_xticks(turns)
        ax.set_xlabel("Conversation turn")
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Conversations flagged by this turn")
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Cumulative detection by turn", fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "10b_detection_by_turn")
    plt.show()
""")

md("""
### Judge scores by turn

The judge's own view of the conversation, rescaled to 0 to 1 so it can be read next to the probe figure in section 7.
""")
code("""
if not judge:
    print("No judge files found, skipping.")
else:
    fig, axes = panels(len(AVAILABLE), sharey=True)
    for ax, m in zip(axes, AVAILABLE):
        ax.set_title(MODELS[m])
        if m not in judge:
            continue
        for lab in ("escalation", "benign_control"):
            sub = judge[m][judge[m].label == lab]
            turns = sorted(sub.turn.unique())
            stats = np.array([boot_ci(sub[sub.turn == t].score.values / 100) for t in turns])
            ax.plot(turns, stats[:, 0], marker="o", color=LABEL_COLORS[lab], label=LABEL_NAMES[lab])
            ax.fill_between(turns, stats[:, 1], stats[:, 2], color=LABEL_COLORS[lab], alpha=0.18, lw=0)
        ax.axhline(0.5, color=GREY, ls=":", lw=1)
        ax.set_xticks(turns)
        ax.set_xlabel("Conversation turn")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Judge score (0 to 100)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "10c_judge_scores")
    plt.show()
    quality = pd.DataFrame({MODELS[m]: judge[m].status.value_counts() for m in AVAILABLE if m in judge}).fillna(0).astype(int)
    print("Judge turn status counts:")
    display(quality)
""")

md("""
## 11. Do the two detectors agree?

Counts of final-turn flags for the probe against the judge, with Cohen's kappa. High agreement on the diagonal with both near the truth means they are redundant. Low agreement means they notice different things, which is where a combined monitor could help.
""")
code("""
if not judge:
    print("No judge files found, skipping.")
else:
    avail_j = [m for m in AVAILABLE if m in judge]
    fig, axes = panels(len(avail_j), width=3.7, height=3.6)
    kappas = {}
    for ax, m in zip(axes, avail_j):
        j = judge[m]
        jf = j[j.turn == j.groupby("record_id").turn.transform("max")].set_index("record_id")["flag"].dropna()
        pf = probe_final.loc[m]["harmful_share"] >= CONVERSATION_SHARE
        ids = pf.index.intersection(jf.index)
        a, b = pf.loc[ids].astype(bool).values, jf.loc[ids].astype(bool).values
        grid = np.array([[np.sum(~a & ~b), np.sum(~a & b)], [np.sum(a & ~b), np.sum(a & b)]])
        ax.imshow(grid, cmap="Blues")
        for (r, c), v in np.ndenumerate(grid):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=14, color="white" if v > grid.max() / 2 else "#111")
        ax.set_xticks([0, 1], ["not flagged", "flagged"])
        ax.set_yticks([0, 1], ["not flagged", "flagged"])
        ax.set_xlabel("Judge")
        ax.grid(False)
        kappas[m] = kappa(a, b)
        ax.set_title(f"{MODELS[m]}\\nkappa = {kappas[m]:.2f}")
    axes[0].set_ylabel("Probe")
    fig.tight_layout()
    save(fig, "11_agreement")
    plt.show()
""")

md("""
## 12. Does the probe signal follow compliance?

If the probe tracks what the model actually produced, the final harmful share should differ between replies you labelled refused, partial and complied. If it does not, the signal reflects the prompt or topic more than the output.
""")
code("""
if not labels:
    print("No compliance label files found, skipping.")
else:
    order = ["refused", "partial", "complied"]
    avail_l = [m for m in AVAILABLE if m in labels]
    fig, axes = panels(len(avail_l), sharey=True)
    for ax, m in zip(axes, avail_l):
        d = full_history[(full_history.model == m) & (full_history.label == "escalation")].merge(labels[m], left_on="record_id", right_on="id", suffixes=("", "_manual"))
        for i, lab in enumerate(order):
            v = d[d.label_manual == lab].harmful_share.values
            ax.scatter(i + jit.uniform(-0.15, 0.15, len(v)), v, s=26, alpha=0.7, color=C_ESC, edgecolor="white", lw=0.4)
            if len(v):
                ax.hlines(v.mean(), i - 0.28, i + 0.28, color="#111", lw=2)
            ax.text(i, 1.03, f"n={len(v)}", ha="center", fontsize=8.5)
        ax.set_xticks(range(3), order)
        ax.set_ylim(-0.02, 1.1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_title(MODELS[m])
    axes[0].set_ylabel("Final reply: tokens flagged harmful")
    fig.suptitle("Probe signal against manual compliance labels (escalation conversations)", fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "12_compliance_vs_signal")
    plt.show()
""")

md("## 13. Summary")
code("""
rows = []
for m in AVAILABLE:
    esc = full_history[(full_history.model == m) & (full_history.label == "escalation")].set_index("record_id").harmful_share
    ctl = full_history[(full_history.model == m) & (full_history.label == "benign_control")].set_index("record_id").harmful_share
    kp = primary[(primary.model == m) & (primary.condition == "crescendo") & (primary.turn == primary.n_turns) & (primary.label == "escalation")].pivot_table(index="record_id", columns="k", values="harmful_share")
    ao = cond_final[(cond_final.model == m) & (cond_final.label == "escalation")].pivot_table(index="record_id", columns="condition", values="harmful_share")
    row = {
        "model": MODELS[m],
        "n esc / ctl": f"{len(esc)} / {len(ctl)}",
        "complied": f"{(labels[m].label == 'complied').mean():.0%}" if m in labels else "n/a",
        "probe test acc": f"{test[(test.model == m) & (test.probe_kind == 'mlp')].token_accuracy.mean():.1%}" if len(test) else "n/a",
        "share esc / ctl": f"{esc.mean():.2f} / {ctl.mean():.2f}",
        "k=1 to k=n (esc)": f"{kp[kp.columns.min()].mean():.2f} to {kp[kp.columns.max()].mean():.2f}",
        "objective minus history": f"{(ao['attack_objective'] - ao['crescendo']).mean():+.2f}",
        "probe AUROC": f"{probe_sep.loc[m, 'auroc']:.2f}",
        "FPR at 0.5": f"{probe_sep.loc[m, 'fpr']:.0%}",
    }
    if judge and m in pj.index:
        row["judge AUROC"] = f"{pj.loc[m, 'judge_auc']:.2f}"
        row["probe-judge kappa"] = f"{kappas[m]:.2f}" if m in kappas else "n/a"
    rows.append(row)
summary = pd.DataFrame(rows).set_index("model")
TABLES.mkdir(parents=True, exist_ok=True)
summary.to_csv(TABLES / "analysis_summary.csv")
summary.T
""")

md("""
### Limits to state alongside any result

- **Small samples.** There are 50 escalation and 50 control conversations per model, from two harm categories. The intervals above show how much that allows.
- **Placeholder objectives.** Prompts are redacted with bracketed placeholders, so replies describe the shape of a phishing or intrusion task, not operational content. Compliance is easier to obtain from small models than in the paper's setting.
- **Different models and attacks.** The paper used Llama-3-8B-Instruct and a circuit-breaker variant with five hand-verified Crescendo attacks. These are small open models with prewritten, non-adaptive conversations, and the masked-response condition was not run. Treat this as a pilot, not an exact replication.
- **One probe recipe.** The probe is trained on public single-turn data, at one selected layer plus one alternate, with greedy decoding and 4-bit weights. Section 9 shows how much the choice of layer and probe matters.
- **One judge.** The black-box baseline is a single judge model with a frozen prompt. A different judge could rank differently.
- **Probe training signal.** The paper notes it is unclear whether such a probe captures harmfulness or some other difference between its training sets. A probe that separates matched controls poorly (section 8) points the same way.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, "notebooks/experiment_results.ipynb")
print("cells:", len(cells))