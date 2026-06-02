# Onboarding prompt — new contributor to the Agentic-PR bug-fix study

Paste this to brief a new collaborator (human or LLM) joining this project. It assumes nothing beyond access to the repo.

---

You are joining an empirical software-engineering study comparing **bug-fix pull requests** authored by AI coding agents against human-authored fixes. Your goal in your first session is to (a) understand the dataset and research questions, (b) reproduce the existing figures, and then (c) be able to add a new research question on request.

## 1. The dataset

- Hosted on HuggingFace: `mabujadallah/GitHub-Agentic-PR-Dataset`.
- Loaded directly from parquet URLs — there is no local download step. URLs are defined at the top of [analysis_utils.py](analysis_utils.py) as `HF_FIX_PRS`, `HF_COMMITS`, `HF_COMMIT_DETAILS`.
- Three tables:
  - **fix_classified_prs** — one row per PR. Filter to `type == 'fix'`, `state == 'closed'`, `agent != 'OpenAI_Codex'`.
  - **fix_pr_commits** — one row per commit, keyed by `pr_id` and `sha`.
  - **fix_pr_commit_details** — one row per file per commit, with `additions` / `deletions`.
- Scope: 4 agents (`Copilot`, `Cursor`, `Claude_Code`, `Devin`) plus a `Human` baseline (`source == 'human'`).
- Coverage: Dec 2024 – Feb 2026 (15 months). The **AIDev cutoff** is 2025-07-31 — the dataset that originally seeded this study (AIDev) only covered up to that date, so we always split results into `AIDev (Dec24-Jul25)` and `Post-AIDev (Aug25-Feb26)`.

Always start a session by calling `load_fix_prs()` from [analysis_utils.py](analysis_utils.py) — it applies the canonical filters and adds the derived columns (`month`, `is_agent`, `is_merged`, `hours_to_merge`, `period`) that the rest of the code expects.

## 2. The research questions

| RQ | Question | Lives in |
|----|----------|----------|
| RQ1 | How does AI bug-fixing **volume** change over time? | theme1 |
| RQ2 | How does bug-fix **acceptance rate** change over time? | theme2 |
| RQ3 | How does **time to merge** change over time? | theme2 |
| RQ4 | How does **patch size** change over time? | theme2 |
| RQ5 | How does **revision burden** change over time? | theme2 |
| RQ6 | Which agent is **best** at bug fixing overall? | theme3 |
| RQ7 | Do developers **switch agents** over time? | theme1 |
| RQ8 | Does performance change **before vs after the AIDev cutoff**? | theme3 |

The themes correspond to the three notebooks ([theme1_adoption_trends.ipynb](theme1_adoption_trends.ipynb), [theme2_quality_over_time.ipynb](theme2_quality_over_time.ipynb), [theme3_agent_comparison.ipynb](theme3_agent_comparison.ipynb)).

## 3. Project conventions you must follow

1. **Single source of truth.** `AGENTS`, `AGENT_COLORS`, `AIDEV_END`, and the `THEME*_DIR` output paths live in [analysis_utils.py](analysis_utils.py). Do not redefine them in a notebook.
2. **Min-N rule.** When aggregating a monthly metric, only report a cell if `n >= 5` — otherwise `None`. This stops a noisy 1- or 2-PR month from dominating the trend lines. Carry this rule into any new monthly RQ.
3. **Stats.**
   - Proportion comparisons (e.g. merge rate) → `chi_square(a_m, a_t, b_m, b_t)`.
   - Continuous distributions (time-to-merge, patch size, revision lines) → `mann_whitney(series_a, series_b)`.
   - Always print significance via `sig_label(p)` (`*`/`**`/`***`/`ns`).
4. **Revision metric.** "Revision effort" excludes the first commit per PR — see `build_revision_stats`. Don't reinvent this counting.
5. **Plots.** Call `set_plot_style()` once per notebook. Use `AGENT_COLORS` for colour. Save with `save_fig(fig, "rqN_slug", THEME{1,2,3}_DIR)`. Annotate monthly charts with the red dotted `2025-07` AIDev cutoff line for consistency.
6. **No duplicate loading code.** If you find yourself writing another `pd.read_parquet(...)`, lift it into [analysis_utils.py](analysis_utils.py) as a helper instead.

## 4. First-session checklist

1. Read [CLAUDE.md](CLAUDE.md) and [analysis_utils.py](analysis_utils.py) end to end.
2. Activate the venv and install deps: `pip install matplotlib seaborn scipy pyarrow fsspec requests pandas`.
3. Run [theme1_adoption_trends.ipynb](theme1_adoption_trends.ipynb) top to bottom — confirm new PNGs appear in [results/theme1_figures/](results/theme1_figures/) that match the committed ones.
4. Open [investigate.py](investigate.py) and run it — this is the ad-hoc diagnostic script for spot-checking dataset anomalies (Jul-25 spike, Copilot revision rate, Devin sparsity). Get comfortable with the patterns it uses.
5. Sketch one paragraph per RQ describing the headline finding from the existing figures. Confirm with the maintainer before drafting any prose for a paper.

## 5. Adding a new RQ

- Decide which theme it belongs in. Don't create a new notebook unless the question is genuinely orthogonal to existing themes.
- Put shared logic in [analysis_utils.py](analysis_utils.py); leave only the narrative + chart cells in the notebook.
- Name the figure `rq{N}_{slug}.png` and save it under the corresponding `THEME*_DIR`.
- Run the relevant notebook end to end and verify the new figure renders before opening a PR.

## 6. Things to ask about before guessing

- Anything that would change the canonical filters in `load_fix_prs()` (e.g. including draft PRs, including Codex, changing the cutoff date) — these affect every figure.
- Anything that would alter `AGENT_COLORS` — these are used in the paper and must stay stable across revisions.
- Whether a new metric should be median or mean — the codebase currently uses median for skewed quantities (lines, hours) and proportions for rates.
