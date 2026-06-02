"""Resumable downloader for the HF parquet files (survives mid-stream aborts)."""
import os, time, urllib.request

BASE = "https://huggingface.co/datasets/mabujadallah/GitHub-Agentic-PR-Dataset/resolve/main"
FILES = ["fix_classified_prs.parquet", "fix_pr_commits.parquet", "fix_pr_commit_details.parquet"]
os.makedirs(".cache", exist_ok=True)


def total_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers.get("Content-Length", 0))


def fetch(name):
    url, dest = f"{BASE}/{name}", f".cache/{name}"
    total = total_size(url)
    for attempt in range(1, 201):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= total:
            print(f"  {name}: complete ({have/1e6:.1f} MB)", flush=True)
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            print(f"  {name}: attempt {attempt} stopped at "
                  f"{os.path.getsize(dest)/1e6:.1f}/{total/1e6:.1f} MB ({type(e).__name__}); resuming",
                  flush=True)
            time.sleep(2)
    raise SystemExit(f"{name}: gave up after 200 attempts")


for n in FILES:
    fetch(n)
print("ALL FILES CACHED", flush=True)
