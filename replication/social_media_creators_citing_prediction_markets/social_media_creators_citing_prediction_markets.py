"""
Replication script for: What Are Social Media Creators Citing From Prediction Markets?

Generates a bar chart showing the distribution of race/contract types referenced
across TikTok + YouTube prediction market videos.

Output: social_media_creators_citing_prediction_markets.png (in this folder)

Usage:
    python replication/social_media_creators_citing_prediction_markets/social_media_creators_citing_prediction_markets.py
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
TT_CSV = BASE_DIR / "final_data" / "tiktok_prediction_markets.csv"
TT_CONTRACT = BASE_DIR / "tiktok" / "data" / "contract_extraction_progress.json"

# YouTube data
YT_CSV = BASE_DIR / "final_data" / "youtube_prediction_markets.csv"
YT_DATA_DIR = BASE_DIR / "youtube" / "data"
YT_CONTRACT_FILES = [
    YT_DATA_DIR / "contract_extraction_progress.json",
    YT_DATA_DIR / "contract_extraction_progress_2.json",
    YT_DATA_DIR / "contract_extraction_progress_3.json",
]

COLORS = {
    "presidential": "#e74c3c",
    "senate": "#3498db",
    "house": "#2ecc71",
    "governor": "#f39c12",
    "policy": "#9b59b6",
    "party_control": "#1abc9c",
    "other": "#95a5a6",
}

RACE_LABELS = {
    "presidential": "Presidential Election",
    "senate": "Senate Races",
    "house": "House Races",
    "governor": "Governor Races",
    "policy": "Policy (tariffs, Fed, regulation)",
    "party_control": "Party Control (House/Senate majority)",
    "other": "Novelty / Other",
}

POLITICAL_RACE_TYPES = ["presidential", "senate", "house", "governor", "policy", "party_control"]


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def load_data():
    """Load videos and contract extraction data from both platforms."""
    videos = {}
    contracts = {}

    # TikTok videos
    with open(TT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("_topic", "").upper() == "YES":
                pid = row["post_id"]
                videos[pid] = {"source": "tiktok"}

    # TikTok contracts
    if TT_CONTRACT.exists():
        with open(TT_CONTRACT, encoding="utf-8") as f:
            tt_contracts = json.load(f)
            for vid, data in tt_contracts.items():
                if "_error" not in data and "error" not in data.get("race_types", []):
                    contracts[vid] = data

    # YouTube videos
    with open(YT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("_topic", "").upper() == "YES":
                vid = row["video_id"]
                videos[vid] = {"source": "youtube"}

    # YouTube contracts
    for yt_file in YT_CONTRACT_FILES:
        if yt_file.exists():
            with open(yt_file, encoding="utf-8") as f:
                yt_contracts = json.load(f)
                for vid, data in yt_contracts.items():
                    if vid not in contracts and "_error" not in data and "error" not in data.get("race_types", []):
                        contracts[vid] = data

    n_tt = sum(1 for v in contracts if v in videos and videos[v]["source"] == "tiktok")
    n_yt = sum(1 for v in contracts if v in videos and videos[v]["source"] == "youtube")
    print(f"Contract extractions: {len(contracts)} (TikTok: {n_tt}, YouTube: {n_yt})")
    return videos, contracts


def chart_race_types(contracts, videos):
    """Generate race type distribution bar chart."""
    counter = Counter()
    for data in contracts.values():
        for rt in data.get("race_types", []):
            if rt in POLITICAL_RACE_TYPES:
                counter[rt] += 1

    sorted_items = counter.most_common()
    raw_labels = [x[0] for x in sorted_items]
    values = [x[1] for x in sorted_items]
    display_labels = [RACE_LABELS.get(l, l) for l in raw_labels]
    colors = [COLORS.get(l, "#95a5a6") for l in raw_labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(display_labels)), values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=25, ha="right", fontsize=10)
    ax.set_title("What Are Social Media Creators Citing From Prediction Markets?",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Videos", fontsize=12)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = OUTPUT_DIR / "social_media_creators_citing_prediction_markets.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    videos, contracts = load_data()
    if not contracts:
        print("ERROR: No contract extraction data found.")
        sys.exit(1)
    chart_race_types(contracts, videos)
