#!/usr/bin/env python3
"""
Statistical significance tests on existing experimental results.

For each environment and method pair, computes:
  - Welch's t-test (unequal variance)
  - Mann-Whitney U test (non-parametric)
  - Cohen's d effect size
  - Bootstrap 95% confidence intervals on mean difference

Outputs LaTeX tables and a summary to stdout.

Usage:
    python scripts/statistical_significance.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIRS = {
    "atari": BASE_DIR / "atari-data",
    "minatar": BASE_DIR / "minatar-data",
    "octax": BASE_DIR / "octax-data",
}

# Comparisons: (method_a, method_b, label)
COMPARISONS = [
    ("qrc-spr-orth", "qrc", r"QRC+SPR+orth vs QRC"),
    ("qrc-spr-orth", "qrc-spr", r"QRC+SPR+orth vs QRC+SPR"),
    ("strq-spr-orth2", "strq", r"StrQ+SPR+orth$^2$ vs StrQ"),
    ("strq-spr-orth2", "strq-spr", r"StrQ+SPR+orth$^2$ vs StrQ+SPR"),
    ("qrc-spr-orth", "dqn-rb1", r"QRC+SPR+orth vs DQN(rb=1)"),
]

LAST_FRAC = 0.10  # fraction of training to use for final return
N_BOOTSTRAP = 10000
ALPHA = 0.05

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def discover_envs(domain: str) -> list[str]:
    """Discover all environment names for a domain."""
    data_dir = DATA_DIRS[domain]
    if not data_dir.exists():
        return []
    envs = set()
    for f in data_dir.glob(f"{domain}_*_*.csv"):
        name = f.stem  # e.g. atari_qrc_Pong
        parts = name.split("_", 2)  # [domain, exp_class, env_id]
        if len(parts) == 3:
            envs.add(parts[2])
    return sorted(envs)


def load_csv(domain: str, exp_class: str, env_id: str) -> pd.DataFrame | None:
    """Load a single CSV file."""
    path = DATA_DIRS[domain] / f"{domain}_{exp_class}_{env_id}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def get_seed_columns(df: pd.DataFrame) -> list[str]:
    """Get columns that contain per-seed data."""
    return [c for c in df.columns if c.startswith("seed")]


def extract_final_returns(domain: str, exp_class: str, env_id: str,
                          last_frac: float = LAST_FRAC) -> np.ndarray:
    """
    Extract per-seed final returns (mean over last `last_frac` of training).
    Returns array of shape (n_seeds,) with one value per seed.
    """
    df = load_csv(domain, exp_class, env_id)
    if df is None:
        return np.array([])

    seed_cols = get_seed_columns(df)
    if not seed_cols:
        return np.array([])

    cutoff = df["step"].max() * (1 - last_frac)
    tail = df[df["step"] >= cutoff]

    per_seed_means = []
    for col in seed_cols:
        vals = tail[col].dropna().values
        if len(vals) > 0:
            per_seed_means.append(float(np.mean(vals)))

    return np.array(per_seed_means)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d effect size (pooled std)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                         / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 alpha: float = ALPHA) -> tuple[float, float]:
    """Bootstrap 95% CI on the difference in means (a - b)."""
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(a_boot) - np.mean(b_boot))
    diffs = np.array(diffs)
    lo = np.percentile(diffs, 100 * alpha / 2)
    hi = np.percentile(diffs, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def run_tests(a: np.ndarray, b: np.ndarray) -> dict:
    """Run all statistical tests. a = proposed method, b = baseline."""
    result = {
        "n_a": len(a), "n_b": len(b),
        "mean_a": np.mean(a) if len(a) > 0 else np.nan,
        "mean_b": np.mean(b) if len(b) > 0 else np.nan,
        "diff": np.nan, "t_stat": np.nan, "t_pval": np.nan,
        "u_stat": np.nan, "u_pval": np.nan,
        "d": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
    }

    if len(a) < 2 or len(b) < 2:
        return result

    result["diff"] = np.mean(a) - np.mean(b)

    # Welch's t-test
    t_stat, t_pval = stats.ttest_ind(a, b, equal_var=False)
    result["t_stat"] = float(t_stat)
    result["t_pval"] = float(t_pval)

    # Mann-Whitney U
    try:
        u_stat, u_pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        result["u_stat"] = float(u_stat)
        result["u_pval"] = float(u_pval)
    except ValueError:
        pass  # identical distributions

    # Cohen's d
    result["d"] = cohens_d(a, b)

    # Bootstrap CI
    ci_lo, ci_hi = bootstrap_ci(a, b)
    result["ci_lo"] = ci_lo
    result["ci_hi"] = ci_hi

    return result


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def compute_aggregate_bootstrap(
    all_a: dict[str, np.ndarray],
    all_b: dict[str, np.ndarray],
    n_boot: int = N_BOOTSTRAP,
) -> dict:
    """
    Stratified bootstrap: for each resample, compute per-game normalized
    score difference, then aggregate via IQM.
    """
    rng = np.random.default_rng(123)
    envs = sorted(set(all_a.keys()) & set(all_b.keys()))
    if not envs:
        return {"iqm_diff": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}

    # Compute per-game normalization ranges
    game_ranges = {}
    for env in envs:
        all_vals = np.concatenate([all_a[env], all_b[env]])
        lo, hi = all_vals.min(), all_vals.max()
        if hi - lo < 1e-12:
            hi = lo + 1
        game_ranges[env] = (lo, hi)

    iqm_diffs = []
    for _ in range(n_boot):
        per_game_diffs = []
        for env in envs:
            a = all_a[env]
            b = all_b[env]
            lo, hi = game_ranges[env]
            a_boot = rng.choice(a, size=len(a), replace=True)
            b_boot = rng.choice(b, size=len(b), replace=True)
            norm_a = (np.mean(a_boot) - lo) / (hi - lo)
            norm_b = (np.mean(b_boot) - lo) / (hi - lo)
            per_game_diffs.append(norm_a - norm_b)
        # IQM of per-game differences
        arr = np.array(per_game_diffs)
        from scipy.stats import trim_mean
        iqm_diffs.append(trim_mean(arr, 0.25))

    iqm_diffs = np.array(iqm_diffs)
    return {
        "iqm_diff": float(np.mean(iqm_diffs)),
        "ci_lo": float(np.percentile(iqm_diffs, 2.5)),
        "ci_hi": float(np.percentile(iqm_diffs, 97.5)),
    }


# ---------------------------------------------------------------------------
# LaTeX output
# ---------------------------------------------------------------------------

def fmt_pval(p: float) -> str:
    if np.isnan(p):
        return r"\textemdash"
    if p < 0.001:
        return r"$<$0.001"
    return f"${p:.3f}$"


def fmt_d(d: float) -> str:
    if np.isnan(d):
        return r"\textemdash"
    return f"${d:+.2f}$"


def fmt_ci(lo: float, hi: float) -> str:
    if np.isnan(lo):
        return r"\textemdash"
    return f"$[{lo:+.1f},\\ {hi:+.1f}]$"


def bold_if_sig(s: str, pval: float) -> str:
    if not np.isnan(pval) and pval < ALPHA:
        return rf"\textbf{{{s}}}"
    return s


def generate_latex_table(
    comparison_label: str,
    method_a: str,
    method_b: str,
    domain: str,
    results: list[tuple[str, dict]],
    aggregate: dict,
) -> str:
    """Generate a LaTeX table for one comparison on one domain."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        rf"\caption{{{comparison_label} on {domain.capitalize()} "
        rf"(final {int(LAST_FRAC*100)}\% of training)}}",
        rf"\label{{tab:sig_{method_a}_{method_b}_{domain}}}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"\textbf{Game} & \textbf{$\Delta$ Mean} & \textbf{Welch $p$} "
        r"& \textbf{M-W $p$} & \textbf{Cohen's $d$} & \textbf{95\% CI} \\",
        r"\midrule",
    ]

    wins, ties, losses = 0, 0, 0
    for env_name, r in results:
        diff_str = f"${r['diff']:+.1f}$" if not np.isnan(r["diff"]) else r"\textemdash"
        t_str = bold_if_sig(fmt_pval(r["t_pval"]), r["t_pval"])
        u_str = bold_if_sig(fmt_pval(r["u_pval"]), r["u_pval"])
        d_str = fmt_d(r["d"])
        ci_str = fmt_ci(r["ci_lo"], r["ci_hi"])

        lines.append(f"{env_name} & {diff_str} & {t_str} & {u_str} & {d_str} & {ci_str} \\\\")

        if not np.isnan(r["t_pval"]):
            if r["t_pval"] < ALPHA and r["diff"] > 0:
                wins += 1
            elif r["t_pval"] < ALPHA and r["diff"] < 0:
                losses += 1
            else:
                ties += 1

    # Summary row
    lines.append(r"\midrule")
    agg_str = (f"IQM $\\Delta$: ${aggregate['iqm_diff']:+.3f}$ "
               f"$[{aggregate['ci_lo']:+.3f},\\ {aggregate['ci_hi']:+.3f}]$")
    lines.append(
        rf"\multicolumn{{6}}{{l}}{{\textbf{{Win/Tie/Loss}}: {wins}/{ties}/{losses} "
        rf"\quad {agg_str}}} \\"
    )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    all_tables: list[str] = []

    for method_a, method_b, label in COMPARISONS:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")

        for domain in ["atari", "minatar", "octax"]:
            envs = discover_envs(domain)
            if not envs:
                continue

            results = []
            all_a_data: dict[str, np.ndarray] = {}
            all_b_data: dict[str, np.ndarray] = {}

            for env_id in envs:
                a = extract_final_returns(domain, method_a, env_id)
                b = extract_final_returns(domain, method_b, env_id)

                if len(a) == 0 and len(b) == 0:
                    continue  # no data for either method

                r = run_tests(a, b)
                env_display = env_id.replace("-v4", "").replace("-v1", "").replace("MinAtar/", "")
                results.append((env_display, r))

                if len(a) > 0 and len(b) > 0:
                    all_a_data[env_id] = a
                    all_b_data[env_id] = b

                # Print summary
                sig_marker = "*" if r["t_pval"] < ALPHA else " "
                direction = "+" if r["diff"] > 0 else "-" if r["diff"] < 0 else "="
                print(f"  {sig_marker} {env_display:20s}  "
                      f"diff={r['diff']:+8.1f}  "
                      f"t-p={r['t_pval']:.4f}  "
                      f"U-p={r['u_pval']:.4f}  "
                      f"d={r['d']:+.2f}  "
                      f"CI=[{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  "
                      f"n=({r['n_a']},{r['n_b']})")

            if not results:
                print(f"  [{domain}] No overlapping data found.")
                continue

            # Aggregate
            aggregate = compute_aggregate_bootstrap(all_a_data, all_b_data)
            wins = sum(1 for _, r in results
                       if not np.isnan(r["t_pval"]) and r["t_pval"] < ALPHA and r["diff"] > 0)
            ties = sum(1 for _, r in results
                       if np.isnan(r["t_pval"]) or r["t_pval"] >= ALPHA)
            losses = sum(1 for _, r in results
                         if not np.isnan(r["t_pval"]) and r["t_pval"] < ALPHA and r["diff"] < 0)
            print(f"\n  [{domain}] Win/Tie/Loss: {wins}/{ties}/{losses}")
            print(f"  [{domain}] Aggregate IQM diff: {aggregate['iqm_diff']:+.4f} "
                  f"CI=[{aggregate['ci_lo']:+.4f}, {aggregate['ci_hi']:+.4f}]")

            table = generate_latex_table(
                label, method_a, method_b, domain, results, aggregate
            )
            all_tables.append(table)

    # Write all tables to file
    output_dir = BASE_DIR / "docs" / "rebuttal_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "statistical_significance.tex"

    tex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{geometry}
\geometry{margin=0.8in}

\begin{document}

\section*{Statistical Significance Tests}

\noindent Per-seed final returns are computed as the mean episodic return over the
last 10\% of training steps. Welch's $t$-test (unequal variance) and
Mann--Whitney $U$ test are applied per game. Cohen's $d$ gives effect size.
Bootstrap 95\% CIs use 10{,}000 resamples. Bold $p$-values indicate
significance at $\alpha = 0.05$.

"""
    tex_content += "\n\n\\bigskip\n\n".join(all_tables)
    tex_content += "\n\n\\end{document}\n"

    tex_path.write_text(tex_content)
    print(f"\n\nLaTeX tables saved to: {tex_path}")


if __name__ == "__main__":
    main()
