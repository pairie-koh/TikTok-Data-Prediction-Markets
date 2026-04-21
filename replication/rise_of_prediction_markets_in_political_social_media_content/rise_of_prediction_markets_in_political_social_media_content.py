"""
Replication script for: The Rise of Prediction Markets in Political Social Media Content

Generates a stacked bar chart showing monthly video counts for polls-referencing
vs prediction-market-referencing videos (TikTok + YouTube combined), with
election year annotations.

Output: rise_of_prediction_markets_in_political_social_media_content.png (in this folder)

Usage:
    python replication/rise_of_prediction_markets_in_political_social_media_content/rise_of_prediction_markets_in_political_social_media_content.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR

# PM datasets (processed)
TIKTOK_PM_CSV = ROOT / "tiktok" / "data" / "tiktok_platform_filtered.csv"
YOUTUBE_PM_CSV = ROOT / "youtube" / "data" / "youtube_platform_filtered.csv"

# Poll datasets
TIKTOK_POLLS_RAW = ROOT / "polls" / "data" / "raw" / "tiktok_polls_raw.json"
YOUTUBE_POLLS_RAW = ROOT / "polls" / "data" / "raw" / "youtube_polls_raw.json"


def load_pm(csv_path, platform, date_col):
    """Load processed PM data, filtered to US-politics videos."""
    if not csv_path.exists():
        print(f"  {platform} PM: NO DATA FOUND at {csv_path}")
        return pd.DataFrame(columns=["date", "source"])

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
    if "_topic" in df.columns:
        df = df[df["_topic"] == "YES"]
    df = df.dropna(subset=["date"])
    print(f"  {platform} PM: {len(df)} political videos")
    return df[["date"]].assign(source="prediction_markets")


def load_polls(json_path, platform, date_col):
    """Load raw poll-referencing video data."""
    if not json_path.exists():
        print(f"  {platform} Polls: NO DATA")
        return pd.DataFrame(columns=["date", "source"])

    data = json.load(open(json_path, encoding="utf-8"))
    df = pd.DataFrame(data)
    if date_col not in df.columns:
        return pd.DataFrame(columns=["date", "source"])

    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"])
    print(f"  {platform} Polls: {len(df)} videos")
    return df[["date"]].assign(source="polls")


def main():
    print("Loading datasets...")

    # Load all data (TikTok + YouTube combined)
    dfs = []
    for loader, path, plat, col in [
        (load_pm, TIKTOK_PM_CSV, "TikTok", "create_time"),
        (load_pm, YOUTUBE_PM_CSV, "YouTube", "date_posted"),
        (load_polls, TIKTOK_POLLS_RAW, "TikTok", "create_time"),
        (load_polls, YOUTUBE_POLLS_RAW, "YouTube", "date_posted"),
    ]:
        d = loader(path, plat, col)
        if len(d) > 0:
            dfs.append(d)

    if not dfs:
        print("ERROR: No data loaded.")
        sys.exit(1)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[(all_df["date"] >= "2020-01-01") & (all_df["date"] <= "2026-12-31")]

    # Monthly counts
    all_df["month"] = all_df["date"].dt.to_period("M").dt.to_timestamp()
    counts = all_df.groupby(["month", "source"]).size().reset_index(name="count")

    # Pivot for plotting
    pivot = counts.pivot_table(index="month", columns="source", values="count", fill_value=0)
    for col in ["polls", "prediction_markets"]:
        if col not in pivot.columns:
            pivot[col] = 0

    months = pivot.index
    bar_width = 20  # days

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot bars - polls behind, PM in front (overlapping style matching original)
    ax.bar(months, pivot["polls"], width=bar_width, color="#3498db", alpha=0.7,
           label="Polls-referencing videos", zorder=2)
    ax.bar(months, pivot["prediction_markets"], width=bar_width, color="#e74c3c", alpha=0.7,
           label="Prediction market videos", zorder=3)

    # Election annotations
    for year, label in [(2020, "2020\nElection"), (2024, "2024\nElection")]:
        election_date = pd.Timestamp(f"{year}-11-01")
        ax.axvline(election_date, color="black", linestyle="--", alpha=0.5, zorder=4)
        ax.text(election_date, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 100,
                label, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
                zorder=5)

    ax.set_title("The Rise of Prediction Markets in Political Social Media Content",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Videos per month (TikTok + YouTube)", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "rise_of_prediction_markets_in_political_social_media_content.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
