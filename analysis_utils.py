"""
analysis_utils.py
Shared utilities for all three theme notebooks.

All notebooks import from here to ensure:
  - Consistent data loading and filtering (load once, reuse)
  - Consistent plot style and colours
  - Reusable statistical tests
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

# ── Data sources ────────────────────────────────────────────────────────────────
# Files are read straight from HuggingFace by default. If a local copy exists under
# the cache dir (CLI_DATA_CACHE, default ".cache"), it is used instead — the
# fix_classified_prs file is ~1.5 GB, so caching avoids re-downloading every run.
# Inert when no cache is present: behaviour is identical to reading from the URL.
_HF = "https://huggingface.co/datasets/mabujadallah/GitHub-Agentic-PR-Dataset/resolve/main"
_CACHE_DIR = Path(os.environ.get("CLI_DATA_CACHE", ".cache"))


def _source(filename: str) -> str:
    """Return the local cached path if present, else the HuggingFace URL."""
    local = _CACHE_DIR / filename
    return str(local) if local.exists() else f"{_HF}/{filename}"


HF_FIX_PRS        = _source("fix_classified_prs.parquet")
HF_COMMITS        = _source("fix_pr_commits.parquet")
HF_COMMIT_DETAILS = _source("fix_pr_commit_details.parquet")

# ── Constants ─────────────────────────────────────────────────────────────────
AGENTS = ["Copilot", "Cursor", "Claude_Code", "Devin"]

# Columns actually used by the analysis. The PR table also carries `body` (full PR
# description text), which for ~2M rows is multiple GB and is never used — reading it
# triggers an out-of-memory on modest machines, so we project to this set instead.
PR_COLUMNS = [
    "id", "number", "state", "created_at", "merged_at",
    "repo_name", "is_agent", "agent", "type", "source",
    "title", "html_url",
]

AGENT_COLORS: dict[str, str] = {
    "Copilot":     "#1f77b4",
    "Cursor":      "#2ca02c",
    "Claude_Code": "#ff7f0e",
    "Devin":       "#d62728",
    "Human":       "#7f7f7f",
}

# AIDev dataset coverage boundary
AIDEV_END = pd.Timestamp("2025-07-31", tz="UTC")

# Minimum sample sizes before a monthly cell is reported (replaces the old n>=5 floor).
# Proportions need more support than medians to stabilise.
MIN_N_PROP   = 30   # monthly merge/revision rates (RQ2, RQ5, RQ7)
MIN_N_MEDIAN = 20   # monthly medians: time-to-merge, patch size (RQ3, RQ4)

# Drop PRs created within this many days of the dataset's most recent created_at:
# recent months are biased toward fast-decision PRs (survivorship), since still-open
# PRs are excluded by the state=='closed' filter.
SURVIVORSHIP_CUTOFF_DAYS = 30

# Public model-release boundaries, for annotating per-agent temporal plots. A single
# named agent spans multiple underlying models over the 15-month window, so within-agent
# trends conflate product evolution with usage shift. Dates are approximate release dates.
MODEL_RELEASES: dict[str, list[tuple[pd.Timestamp, str]]] = {
    "Claude_Code": [
        (pd.Timestamp("2025-02-24", tz="UTC"), "Sonnet 3.7"),
        (pd.Timestamp("2025-05-22", tz="UTC"), "Sonnet/Opus 4"),
        (pd.Timestamp("2025-09-29", tz="UTC"), "Sonnet 4.5"),
    ],
}

# Lazily-computed set of repos covered by the AIDev period (created_at <= AIDEV_END).
# Cached on the module after first computation. Used to hold RQ8's temporal comparison
# to a fixed repo set rather than a shifting denominator.
_AIDEV_REPOS: set[str] | None = None

# Output directories
RESULTS_DIR = Path("results")
THEME1_DIR  = RESULTS_DIR / "theme1_figures"
THEME2_DIR  = RESULTS_DIR / "theme2_figures"
THEME3_DIR  = RESULTS_DIR / "theme3_figures"

for _d in [THEME1_DIR, THEME2_DIR, THEME3_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_fix_prs(
    *,
    restrict_to_aidev_repos: bool = False,
    apply_survivorship_cutoff: bool = True,
) -> pd.DataFrame:
    """
    Load and preprocess bug-fix PRs from HuggingFace.

    Filters to: type='fix', state='closed', agent != 'OpenAI_Codex'.
    Adds derived columns: repo (=repo_name), month, is_agent, is_merged,
    hours_to_merge, period.

    Parameters
    ----------
    restrict_to_aidev_repos:
        When True, keep only PRs whose repo appears in the AIDev coverage set
        (repos seen at or before AIDEV_END). Required for RQ8 so the temporal
        comparison is on a fixed repo set, not a shifting denominator.
    apply_survivorship_cutoff:
        When True (default), drop PRs created within SURVIVORSHIP_CUTOFF_DAYS of
        the most recent created_at. Recent months are otherwise biased toward
        fast-decision PRs, since still-open PRs are excluded by state=='closed'.
    """
    global _AIDEV_REPOS
    print("Loading fix PRs from HuggingFace ...")
    df = pd.read_parquet(
        HF_FIX_PRS,
        columns=PR_COLUMNS,
        filters=[
            ("type",  "==", "fix"),
            ("state", "==", "closed"),
            ("agent", "!=", "OpenAI_Codex"),
        ],
    )

    df["created_at"]     = pd.to_datetime(df["created_at"], utc=True)
    df["merged_at"]      = pd.to_datetime(df["merged_at"],  utc=True, errors="coerce")
    df["repo"]           = df["repo_name"]
    df["month"]          = df["created_at"].dt.to_period("M")
    df["is_agent"]       = df["source"] == "agent"
    df["is_merged"]      = df["merged_at"].notna()
    df["hours_to_merge"] = (
        (df["merged_at"] - df["created_at"]).dt.total_seconds() / 3600
    ).where(df["is_merged"])
    df["period"] = df["created_at"].apply(
        lambda x: "AIDev (Dec24-Jul25)" if x <= AIDEV_END else "Post-AIDev (Aug25-Feb26)"
    )

    # Cache the AIDev repo coverage on first load (computed on the full, unrestricted set).
    if _AIDEV_REPOS is None:
        _AIDEV_REPOS = set(df.loc[df["created_at"] <= AIDEV_END, "repo"].dropna().unique())
        print(f"  AIDev repo coverage: {len(_AIDEV_REPOS):,} distinct repos")

    if apply_survivorship_cutoff:
        cutoff = df["created_at"].max() - pd.Timedelta(days=SURVIVORSHIP_CUTOFF_DAYS)
        before = len(df)
        df = df[df["created_at"] <= cutoff].copy()
        print(f"  Survivorship cutoff at {cutoff.date()}: dropped {before - len(df):,} recent PRs")

    if restrict_to_aidev_repos:
        before = len(df)
        df = df[df["repo"].isin(_AIDEV_REPOS)].copy()
        print(f"  Restricted to AIDev repos: kept {len(df):,} of {before:,} PRs")

    n_agent = int(df["is_agent"].sum())
    n_human = int((~df["is_agent"]).sum())
    print(f"  Fix PRs loaded: {len(df):,}  |  Agent: {n_agent:,}  |  Human: {n_human:,}")
    return df


def load_commits() -> pd.DataFrame:
    """Load per-PR commit metadata."""
    print("Loading commits from HuggingFace ...")
    df = pd.read_parquet(HF_COMMITS)
    print(f"  Commits loaded: {len(df):,}")
    return df


def load_commit_details() -> pd.DataFrame:
    """Load per-file line-change details per commit."""
    print("Loading commit details from HuggingFace ...")
    df = pd.read_parquet(
        HF_COMMIT_DETAILS,
        columns=["sha", "pr_id", "filename", "additions", "deletions"],
    )
    print(f"  Commit details loaded: {len(df):,}")
    return df


def build_revision_stats(
    fix_prs: pd.DataFrame,
    commits: pd.DataFrame,
    details: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every merged PR, compute:
      num_commits        – total commits on the PR
      rev_lines_added    – lines added in revision commits (all except the first)
      rev_lines_deleted  – lines deleted in revision commits

    Returns one row per merged PR, with agent/source/month/period/is_agent metadata.
    """
    merged_ids = set(fix_prs.loc[fix_prs["is_merged"], "id"])

    pr_commits = commits[commits["pr_id"].isin(merged_ids)].copy()
    cpr = pr_commits.groupby("pr_id").size().reset_index(name="num_commits")

    # Exclude the first commit per PR when calculating revision effort
    first_shas = set(pr_commits.groupby("pr_id")["sha"].first().values)
    rev_shas   = set(pr_commits["sha"]) - first_shas

    rev_agg = (
        details[details["sha"].isin(rev_shas)]
        .groupby("pr_id")
        .agg(rev_lines_added=("additions", "sum"), rev_lines_deleted=("deletions", "sum"))
        .reset_index()
    )

    stats = cpr.merge(rev_agg, on="pr_id", how="left").fillna(0)

    meta = (
        fix_prs[["id", "agent", "source", "month", "period", "is_agent"]]
        .rename(columns={"id": "pr_id"})
    )
    return stats.merge(meta, on="pr_id", how="left")


# ── Statistical Tests ─────────────────────────────────────────────────────────

def merge_rate(df: pd.DataFrame) -> tuple[int, int, float]:
    """Return (n_merged, n_total, rate_pct)."""
    n_merged = int(df["is_merged"].sum())
    n_total  = len(df)
    rate     = n_merged / n_total * 100 if n_total else 0.0
    return n_merged, n_total, rate


def chi_square(a_m: int, a_t: int, b_m: int, b_t: int) -> tuple[float, float]:
    """Chi-square test of independence on two merge-rate groups."""
    table         = np.array([[a_m, a_t - a_m], [b_m, b_t - b_m]])
    chi2, p, _, _ = sp_stats.chi2_contingency(table)
    return float(chi2), float(p)


def mann_whitney(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test."""
    u, p = sp_stats.mannwhitneyu(a.dropna(), b.dropna(), alternative="two-sided")
    return float(u), float(p)


def sig_label(p: float) -> str:
    """
    Return significance stars. Pass an ADJUSTED p-value (see bh_correct) — with
    sample sizes in the thousands, raw p-values are almost always significant.
    Always report an effect size (odds_ratio_ci / cliffs_delta) alongside this.
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# ── Effect sizes ──────────────────────────────────────────────────────────────

def odds_ratio_ci(
    a_m: int, a_t: int, b_m: int, b_t: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """
    Odds ratio of group A's success odds vs group B's, with a Wald CI.

    Companion to chi_square for proportion comparisons (e.g. merge rate).
    Uses a Haldane-Anscombe 0.5 continuity correction so empty cells don't blow up.
    Returns (odds_ratio, ci_low, ci_high). OR > 1 => A has higher odds than B.
    """
    a_f, b_f = a_t - a_m, b_t - b_m
    # Haldane-Anscombe correction
    a_m_c, a_f_c, b_m_c, b_f_c = a_m + 0.5, a_f + 0.5, b_m + 0.5, b_f + 0.5
    or_ = (a_m_c / a_f_c) / (b_m_c / b_f_c)
    se  = np.sqrt(1 / a_m_c + 1 / a_f_c + 1 / b_m_c + 1 / b_f_c)
    z   = sp_stats.norm.ppf(1 - alpha / 2)
    lo  = float(np.exp(np.log(or_) - z * se))
    hi  = float(np.exp(np.log(or_) + z * se))
    return float(or_), lo, hi


def cliffs_delta(a: pd.Series, b: pd.Series) -> tuple[float, str]:
    """
    Cliff's delta, a nonparametric effect size for two distributions.

    Companion to mann_whitney. delta in [-1, 1]; sign follows a vs b
    (delta > 0 => a tends to be larger). Magnitude thresholds per Romano et al.:
    |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    a = a.dropna().to_numpy()
    b = b.dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return float("nan"), "n/a"
    # Rank-based O(n log n) computation rather than the O(n*m) double loop.
    u, _ = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    delta = float(2 * u / (len(a) * len(b)) - 1)
    ad = abs(delta)
    mag = ("negligible" if ad < 0.147 else
           "small"      if ad < 0.33  else
           "medium"     if ad < 0.474 else
           "large")
    return delta, mag


def bh_correct(pvals: list[float]) -> list[float]:
    """
    Benjamini-Hochberg FDR correction. Returns adjusted p-values in the input order.
    Apply across each family of tests within an RQ before calling sig_label.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return []
    order  = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest rank downward
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    return adj.tolist()


# ── Plot Helpers ──────────────────────────────────────────────────────────────

def set_plot_style() -> None:
    """Apply consistent visual style across all themes."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":        120,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def save_fig(fig: plt.Figure, name: str, folder: Path) -> Path:
    """Save figure as PNG and close it."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Saved: {path}")
    return path


def annotate_model_releases(ax, agent: str, month_index: list[str]) -> None:
    """
    Draw thin grey vertical lines at known model-release boundaries for `agent`,
    so within-agent temporal trends can be read against product changes.
    `month_index` is the list of 'YYYY-MM' strings on the x-axis.
    """
    for ts, label in MODEL_RELEASES.get(agent, []):
        m = f"{ts.year:04d}-{ts.month:02d}"
        if m in month_index:
            x = month_index.index(m)
            ax.axvline(x, color="grey", linestyle="-", linewidth=0.8, alpha=0.5)
            ax.text(x, ax.get_ylim()[1], label, rotation=90, va="top", ha="right",
                    fontsize=7, color="grey", alpha=0.8)


# ── Confidence Intervals ────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion, returned as percentages.
    More reliable than the normal approximation at small n / extreme rates.
    Returns (lo_pct, hi_pct); (nan, nan) if n == 0.
    """
    if n == 0:
        return float("nan"), float("nan")
    z   = sp_stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half   = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return float((centre - half) * 100), float((centre + half) * 100)


def bootstrap_median_ci(
    values: pd.Series, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for a median. Returns (lo, hi); (nan, nan) if empty.
    """
    v = values.dropna().to_numpy()
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.median(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


# ── File-role classification (RQ4) ──────────────────────────────────────────────

_TEST_RE = re.compile(
    r"(^|/)tests?/|_test\.|test_|\.test\.|\.spec\.|/__tests__/", re.IGNORECASE
)
_GEN_RE = re.compile(
    r"\.lock$|lock\.(json|ya?ml)$|(^|/)go\.sum$|\.min\.|"
    r"(^|/)(dist|build|vendor|node_modules)/|"
    r"__generated__|\.generated\.|\.snap$|\.pb\.go$",
    re.IGNORECASE,
)


def classify_file_role(filename: str) -> str:
    """
    Classify a changed file as 'test', 'generated', or 'prod'.
    Patch-size analysis (RQ4) should report these separately: agents and humans
    write different amounts of test and generated code, so a single lines-added
    number is hard to interpret.
    """
    if not isinstance(filename, str):
        return "prod"
    if _GEN_RE.search(filename):
        return "generated"
    if _TEST_RE.search(filename):
        return "test"
    return "prod"


# ── Repo-matched human baseline (RQ2/RQ3/RQ4/RQ6) ────────────────────────────────

def build_matched_human_baseline(
    fix_prs: pd.DataFrame, k: int = 3, window_days: int = 30, seed: int = 0
) -> pd.DataFrame:
    """
    Build a repo- and time-matched comparison set so agent-vs-human differences
    are not confounded by repo mix (agents cluster in certain repos).

    For each agent PR, sample up to k human PRs from the SAME repo whose created_at
    is within +/-window_days. Returns the agent PRs plus their matched human PRs
    (same schema as fix_prs); unmatched human PRs are dropped.
    """
    rng = np.random.default_rng(seed)
    agents = fix_prs[fix_prs["is_agent"]].copy()
    humans = fix_prs[~fix_prs["is_agent"]].copy()
    window = pd.Timedelta(days=window_days)

    humans_by_repo = {repo: g for repo, g in humans.groupby("repo")}
    matched_idx: set = set()
    for _, pr in agents.iterrows():
        pool = humans_by_repo.get(pr["repo"])
        if pool is None:
            continue
        lo, hi = pr["created_at"] - window, pr["created_at"] + window
        cand = pool.index[(pool["created_at"] >= lo) & (pool["created_at"] <= hi)]
        if len(cand) == 0:
            continue
        take = cand if len(cand) <= k else rng.choice(cand, size=k, replace=False)
        matched_idx.update(np.atleast_1d(take).tolist())

    matched_humans = humans.loc[sorted(matched_idx)]
    out = pd.concat([agents, matched_humans], ignore_index=True)
    print(f"  Matched baseline: {len(agents):,} agent PRs + "
          f"{len(matched_humans):,} matched human PRs "
          f"(from {len(humans):,} human PRs)")
    return out
