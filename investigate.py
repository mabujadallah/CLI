import pandas as pd
import warnings
warnings.filterwarnings("ignore")

_HF = "https://huggingface.co/datasets/mabujadallah/GitHub-Agentic-PR-Dataset/resolve/main"
HF_FIX_PRS = f"{_HF}/fix_classified_prs.parquet"
HF_COMMITS  = f"{_HF}/fix_pr_commits.parquet"
AGENTS = ["Copilot", "Cursor", "Claude_Code", "Devin"]

print("Loading fix PRs...")
df = pd.read_parquet(
    HF_FIX_PRS,
    filters=[("type","==","fix"),("state","==","closed"),("agent","!=","OpenAI_Codex")]
)
df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
df["merged_at"]  = pd.to_datetime(df["merged_at"], utc=True, errors="coerce")
df["is_agent"]   = df["source"] == "agent"
df["is_merged"]  = df["merged_at"].notna()
df["month"]      = df["created_at"].dt.to_period("M")

# ── ISSUE 3: Jul-25 spike ─────────────────────────────────────────────
print("\n=== ISSUE 3: Jul-25 agent breakdown ===")
jul = df[df["month"] == "2025-07"]
print(jul.groupby("agent").size().sort_values(ascending=False))
total_per_agent = df.groupby("agent").size()
jul_per_agent   = jul.groupby("agent").size()
share = (jul_per_agent / total_per_agent * 100).round(1).sort_values(ascending=False)
print("\nJul-25 as % of each agent's total PRs:")
print(share)

# ── ISSUE 1: Copilot revision rate ───────────────────────────────────
print("\n=== ISSUE 1: Revision rate per agent (num_commits > 1) ===")
print("Loading commits...")
commits = pd.read_parquet(HF_COMMITS)
merged_ids = set(df.loc[df["is_merged"], "id"])
pr_commits = commits[commits["pr_id"].isin(merged_ids)].copy()
ncommits = pr_commits.groupby("pr_id").size().reset_index(name="num_commits")
meta = df[["id", "agent", "source"]].rename(columns={"id": "pr_id"})
ncommits = ncommits.merge(meta, on="pr_id", how="left")

print(f"\n{'Agent':<14} {'Revised%':>10} {'Median commits':>16} {'N':>8}")
print("-" * 52)
for agent in AGENTS + ["human"]:
    if agent == "human":
        sub = ncommits[ncommits["source"] == "human"]
    else:
        sub = ncommits[ncommits["agent"] == agent]
    if len(sub) == 0:
        continue
    revised = (sub["num_commits"] > 1).mean() * 100
    med = sub["num_commits"].median()
    print(f"{agent:<14} {revised:>9.1f}% {med:>15.1f}  {len(sub):>8,}")

print("\nCopilot num_commits distribution:")
cop = ncommits[ncommits["agent"] == "Copilot"]["num_commits"]
print(cop.describe())
print("Value counts (top 10):")
print(cop.value_counts().head(10))

# ── ISSUE 2: Devin sparsity ───────────────────────────────────────────
print("\n=== ISSUE 2: Devin monthly PR count ===")
devin = df[df["agent"] == "Devin"]
monthly = devin.groupby("month").agg(total=("id","count"), merged=("is_merged","sum")).tail(10)
monthly["merge_rate"] = (monthly["merged"] / monthly["total"] * 100).round(1)
print(monthly)
