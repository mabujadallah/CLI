# Agentic Bug-Fix PR Study

An empirical study comparing **bug-fix pull requests** authored by AI coding agents (Copilot, Cursor, Claude Code, Devin) against human-authored fixes on GitHub. The analysis answers 8 research questions across 3 themes, covering 15 months of data (Dec 2024 – Feb 2026).

---

## Dataset

- **Source:** [`mabujadallah/GitHub-Agentic-PR-Dataset`](https://huggingface.co/datasets/mabujadallah/GitHub-Agentic-PR-Dataset) on HuggingFace
- **Scope:** Closed bug-fix PRs (`type == 'fix'`, `state == 'closed'`), excluding OpenAI Codex
- **Coverage:** Dec 2024 – Feb 2026 (15 months, single uniform collection pipeline)
- **AIDev cutoff:** 2025-07-31 — splits the timeline into AIDev vs post-AIDev periods (RQ8)

---

## Research Questions

| Theme | RQ | Description |
|---|---|---|
| 1 — Adoption Trends | RQ1 | Monthly volume of agent vs human bug-fix PRs |
| 1 — Adoption Trends | RQ7 | Agent market share over time |
| 2 — Quality Over Time | RQ2 | Merge rate trends |
| 2 — Quality Over Time | RQ3 | Time-to-merge trends |
| 2 — Quality Over Time | RQ4 | Patch size by file role (prod / test / generated) |
| 2 — Quality Over Time | RQ5 | Revision burden (commit count & revision lines) |
| 3 — Agent Comparison | RQ6 | Best agent overall (merge rate, TTM, patch size, revision) |
| 3 — Agent Comparison | RQ8 | AIDev vs post-AIDev period comparison |

---

## Repository Layout

```
├── analysis_utils.py               # Shared helpers: data loading, stats, plotting
├── theme1_adoption_trends.ipynb    # RQ1, RQ7
├── theme2_quality_over_time.ipynb  # RQ2, RQ3, RQ4, RQ5
├── theme3_agent_comparison.ipynb   # RQ6, RQ8
├── results/
│   ├── theme1_figures/             # PNG figures for theme 1
│   ├── theme2_figures/             # PNG figures for theme 2
│   └── theme3_figures/             # PNG figures for theme 3
└── .cache/                         # Local parquet cache (git-ignored, ~1.5 GB)
```

---

## Getting Started

### Prerequisites

```bash
pip install matplotlib seaborn scipy pyarrow fsspec requests pandas jupyter
```

### Running the Analysis

```bash
# from repo root (activate your venv first)
jupyter nbconvert --to notebook --execute theme1_adoption_trends.ipynb --inplace
jupyter nbconvert --to notebook --execute theme2_quality_over_time.ipynb --inplace
jupyter nbconvert --to notebook --execute theme3_agent_comparison.ipynb --inplace
```

Figures are saved to `results/theme{1,2,3}_figures/` as 150 DPI PNGs.

### Local Data Cache

The dataset (`fix_classified_prs.parquet`) is ~1.5 GB and is downloaded automatically from HuggingFace on first run. To avoid re-downloading, place the parquet files in `.cache/`:

```
.cache/
├── fix_classified_prs.parquet
├── fix_pr_commits.parquet
└── fix_pr_commit_details.parquet
```

---

## Methodology Notes

- **Minimum support:** ≥30 PRs per monthly cell for rates; ≥20 for medians (cells below threshold are dropped).
- **Confidence intervals:** Wilson CI for proportions; bootstrap CI for medians — shown as shaded bands on all trend charts.
- **Statistical tests:** Chi-square + odds ratio (proportions); Mann-Whitney U + Cliff's delta (continuous). P-values are FDR-corrected (Benjamini-Hochberg) within each family.
- **Repo-matched baseline:** Agent-vs-human comparisons include a repo-matched human baseline to control for repo-mix confounding.
- **Survivorship bias:** PRs from the last 30 days of the collection window are dropped.

### Known Data Limitations

- No auto-merge signal (`auto_merge`, `merged_by`, `review_comments_count` absent) — merge rate and TTM partly reflect repo merge policy.
- No review-comment counts — revision burden uses commit count and revision lines only.
- `type == 'fix'` label is unvalidated; `results/label_validation_sample.csv` is provided for manual spot-checking.

---

## Agents Studied

| Agent | Color |
|---|---|
| GitHub Copilot | `#6e40c9` |
| Cursor | `#0ea5e9` |
| Claude Code | `#d97706` |
| Devin | `#16a34a` |
| Human (baseline) | `#6b7280` |
