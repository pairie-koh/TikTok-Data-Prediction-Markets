"""
Replication script for: Top 20 Candidates/Politicians Mentioned

Generates a horizontal bar chart of the top 20 candidates/politicians mentioned
across TikTok + YouTube prediction market videos.

Output: top_20_candidates_mentioned.png (in this folder)

Usage:
    python replication/top_20_candidates_mentioned/top_20_candidates_mentioned.py
"""

import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

csv.field_size_limit(10_000_000)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(exist_ok=True)

# TikTok data
TT_DATA_DIR = BASE_DIR / "tiktok" / "data"
TT_CONTRACT = TT_DATA_DIR / "contract_extraction_progress.json"

# YouTube data
YT_DATA_DIR = BASE_DIR / "youtube" / "data"
YT_CONTRACT_FILES = [
    YT_DATA_DIR / "contract_extraction_progress.json",
    YT_DATA_DIR / "contract_extraction_progress_2.json",
    YT_DATA_DIR / "contract_extraction_progress_3.json",
]


def load_contracts():
    """Load and merge TikTok + YouTube contract extraction data."""
    contracts = {}

    # TikTok
    if TT_CONTRACT.exists():
        with open(TT_CONTRACT, encoding="utf-8") as f:
            tt_contracts = json.load(f)
            for vid, data in tt_contracts.items():
                if "_error" not in data and "error" not in data.get("race_types", []):
                    contracts[vid] = data

    # YouTube
    for yt_file in YT_CONTRACT_FILES:
        if yt_file.exists():
            with open(yt_file, encoding="utf-8") as f:
                yt_contracts = json.load(f)
                for vid, data in yt_contracts.items():
                    if vid not in contracts and "_error" not in data and "error" not in data.get("race_types", []):
                        contracts[vid] = data

    print(f"Total contract extractions (no errors): {len(contracts)}")
    return contracts


def chart_candidates(contracts):
    """Generate top 20 candidates/politicians mentioned horizontal bar chart."""
    NORMALIZE = {
        "Donald Trump": "Trump", "donald trump": "Trump",
        "Kamala Harris": "Harris", "kamala harris": "Harris", "Kamala": "Harris",
        "Joe Biden": "Biden", "joe biden": "Biden",
        "J.D. Vance": "Vance", "JD Vance": "Vance",
        "Ron DeSantis": "DeSantis", "ron desantis": "DeSantis",
        "Vivek Ramaswamy": "Ramaswamy", "vivek ramaswamy": "Ramaswamy",
        "Nikki Haley": "Haley", "nikki haley": "Haley",
        "Robert F. Kennedy Jr.": "RFK Jr.", "Robert Kennedy": "RFK Jr.",
        "RFK Jr": "RFK Jr.", "Bobby Kennedy": "RFK Jr.",
        "Gavin Newsom": "Newsom", "gavin newsom": "Newsom",
        "Tim Walz": "Walz",
        "Barack Obama": "Obama", "Michelle Obama": "Michelle Obama",
        "Nancy Pelosi": "Pelosi",
        "Elon Musk": "Musk",
    }

    counter = Counter()
    for data in contracts.values():
        for c in data.get("candidates_mentioned", []):
            normalized = NORMALIZE.get(c, c)
            counter[normalized] += 1

    top = counter.most_common(20)
    labels = [x[0] for x in top][::-1]
    values = [x[1] for x in top][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(labels, values, color="#e74c3c", edgecolor="white", linewidth=0.5)
    ax.set_title("Top 20 Candidates/Politicians Mentioned", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Videos", fontsize=12)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(val), ha="left", va="center", fontsize=10)

    plt.tight_layout()
    path = OUTPUT_DIR / "top_20_candidates_mentioned.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    contracts = load_contracts()
    if not contracts:
        print("ERROR: No contract extraction data found.")
        sys.exit(1)
    chart_candidates(contracts)
