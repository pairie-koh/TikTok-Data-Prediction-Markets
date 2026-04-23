"""
Total views over time: Prediction Market vs Polling videos.

Generates a line chart showing monthly total views (TikTok + YouTube combined)
for prediction market and polling videos, 2020-2026.

Output: total_views_pm_vs_polls.png (in this folder)

Usage:
    python trends/total_views.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

csv.field_size_limit(10_000_000)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

def safe_int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def main():
    monthly = defaultdict(lambda: {"pm_views": 0, "polls_views": 0})

    for path, content, view_col, date_col in [
        (ROOT / "final_data/tiktok_prediction_markets.csv", "pm", "play_count", "create_time"),
        (ROOT / "final_data/youtube_prediction_markets.csv", "pm", "views", "date_posted"),
        (ROOT / "final_data/tiktok_polls.csv", "polls", "play_count", "create_time"),
        (ROOT / "final_data/youtube_polls.csv", "polls", "views", "date_posted"),
    ]:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                date = row.get(date_col, "")[:7]
                if len(date) != 7:
                    continue
                year = int(date[:4])
                if year < 2020 or year > 2026:
                    continue
                views = safe_int(row.get(view_col, 0))
                monthly[date][f"{content}_views"] += views

    months = sorted(monthly.keys())
    pm_views = np.array([monthly[m]["pm_views"] for m in months])
    polls_views = np.array([monthly[m]["polls_views"] for m in months])
    dates = [pd.Timestamp(m + "-01") for m in months]

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(dates, polls_views / 1e6, color="#3498db", linewidth=2.5,
            label="Polls-referencing videos", marker="o", markersize=3, zorder=3)
    ax.plot(dates, pm_views / 1e6, color="#e74c3c", linewidth=2.5,
            label="Prediction market videos", marker="o", markersize=3, zorder=3)

    for year, label in [(2020, "2020\nElection"), (2024, "2024\nElection")]:
        election_date = pd.Timestamp(f"{year}-11-01")
        ax.axvline(election_date, color="black", linestyle="--", alpha=0.4, zorder=1)

    ax.set_title("Total Views: Prediction Market vs Polling Videos Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Total views per month, millions (TikTok + YouTube)", fontsize=12)
    ax.legend(fontsize=11, loc="upper left")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(labelsize=10)

    ax.grid(True, axis="y", alpha=0.3)
    ax.fill_between(dates, pm_views / 1e6, alpha=0.1, color="#e74c3c", zorder=2)
    ax.fill_between(dates, polls_views / 1e6, alpha=0.1, color="#3498db", zorder=2)

    plt.tight_layout()
    outpath = SCRIPT_DIR / "total_views_pm_vs_polls.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
