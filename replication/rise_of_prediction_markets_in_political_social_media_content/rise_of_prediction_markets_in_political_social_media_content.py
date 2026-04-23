"""
Replication script for: The Rise of Prediction Markets in Political Social Media Content

Generates a stacked bar chart showing monthly video counts for polls-referencing
vs prediction-market-referencing videos (TikTok + YouTube combined), with
election year annotations.

Output: rise_of_prediction_markets_in_political_social_media_content.png (in this folder)

Usage:
    python replication/rise_of_prediction_markets_in_political_social_media_content/rise_of_prediction_markets_in_political_social_media_content.py
"""

import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

csv.field_size_limit(10_000_000)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR

# PM datasets (filtered to political content)
TIKTOK_PM_CSV = ROOT / "final_data" / "tiktok_prediction_markets.csv"
YOUTUBE_PM_CSV = ROOT / "final_data" / "youtube_prediction_markets.csv"

# Poll datasets (filtered to political content)
TIKTOK_POLLS_CSV = ROOT / "final_data" / "tiktok_polls.csv"
YOUTUBE_POLLS_CSV = ROOT / "final_data" / "youtube_polls.csv"


_AFFILIATE_RE = re.compile(
    r"kalshi\.pxf\.io|polymarket\.com/ref|predictit.*ref|\.ly/|affiliate|referral",
    re.IGNORECASE,
)


def _is_affiliate_only(description: str) -> bool:
    """True if the PM reference in the description is just an affiliate link."""
    return bool(_AFFILIATE_RE.search(description))


def load_filtered_csv(csv_path, platform, date_col, source_label):
    """Load filtered CSV, keeping only _topic=YES rows.

    For prediction_markets videos with description-only matches, exclude rows
    where the PM reference is just an affiliate link or bare URL (e.g. Kalshi
    affiliate links retroactively added to old video descriptions). Videos with
    genuine text mentions in the description are kept.
    """
    if not csv_path.exists():
        print(f"  {platform} {source_label}: NO DATA FOUND at {csv_path}")
        return pd.DataFrame(columns=["date", "source"])

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
    if "_topic" in df.columns:
        df = df[df["_topic"] == "YES"]
    df = df.dropna(subset=["date"])

    if source_label == "prediction_markets" and "_match_source" in df.columns:
        before = len(df)
        # Only filter description-only matches; keep transcript/title matches as-is
        desc_only = ~df["_match_source"].str.contains("transcript|title", case=False, na=False)
        affiliate_mask = desc_only & df["description"].apply(
            lambda d: _is_affiliate_only(str(d)) if pd.notna(d) else True
        )
        df = df[~affiliate_mask]
        removed = before - len(df)
        print(f"  {platform} {source_label}: {len(df)} political videos ({removed} affiliate/URL-only removed)")
    else:
        print(f"  {platform} {source_label}: {len(df)} political videos")

    return df[["date"]].assign(source=source_label)


def main():
    print("Loading datasets...")

    # Load all data (TikTok + YouTube combined), all filtered to _topic=YES
    dfs = []
    for path, plat, col, source in [
        (TIKTOK_PM_CSV, "TikTok", "create_time", "prediction_markets"),
        (YOUTUBE_PM_CSV, "YouTube", "date_posted", "prediction_markets"),
        (TIKTOK_POLLS_CSV, "TikTok", "create_time", "polls"),
        (YOUTUBE_POLLS_CSV, "YouTube", "date_posted", "polls"),
    ]:
        d = load_filtered_csv(path, plat, col, source)
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
