"""
Polls vs. Prediction Markets: Over-Time Citation Comparison.

Compares how often people cite polls vs. prediction markets when
discussing politics on TikTok and YouTube over time.

Loads from:
  tiktok/data/tiktok_platform_filtered.csv   (PM, 593 political)
  youtube/data/youtube_platform_filtered.csv  (PM, 4078 political)
  polls/data/raw/tiktok_polls_raw.json        (polls, to be collected)
  polls/data/raw/youtube_polls_raw.json       (polls, to be collected)

Produces:
  output/polls_vs_pm_temporal.png  (4-panel figure)
  stdout summary statistics

Usage:
    cd TikTok-Data-Prediction-Markets
    python scripts/analysis_polls_vs_pm.py
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

# ---------------------------------------------------------------------------
# Paths — all relative to repo root
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

# PM datasets (processed)
TIKTOK_PM_CSV = ROOT / "tiktok" / "data" / "tiktok_platform_filtered.csv"
TIKTOK_PM_INFO = ROOT / "tiktok" / "data" / "info_vs_trading_progress.json"
YOUTUBE_PM_CSV = ROOT / "youtube" / "data" / "youtube_platform_filtered.csv"
YOUTUBE_PM_INFO = ROOT / "youtube" / "data" / "info_vs_trading_progress.json"

# Poll datasets (collected via polls/scripts/polls_collection.py)
TIKTOK_POLLS_RAW = ROOT / "polls" / "data" / "raw" / "tiktok_polls_raw.json"
YOUTUBE_POLLS_RAW = ROOT / "polls" / "data" / "raw" / "youtube_polls_raw.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pm(csv_path: Path, info_path: Path, platform: str, date_col: str) -> pd.DataFrame:
    """Load processed PM data, filtered to US-politics videos."""
    if not csv_path.exists():
        print(f"  {platform} PM: NO DATA FOUND at {csv_path}")
        return pd.DataFrame(columns=["date", "source", "platform"])

    df = pd.read_csv(csv_path)
    total = len(df)
    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)

    # Filter to US-politics only
    if "_topic" in df.columns:
        df = df[df["_topic"] == "YES"]

    df = df.dropna(subset=["date"])

    # Report info/trading split
    if info_path.exists():
        info_data = json.load(open(info_path, encoding="utf-8"))
        from collections import Counter
        counts = Counter(info_data.values())
        print(f"  {platform} PM (processed): {len(df)} political (from {total:,} keyword-filtered)")
        print(f"    INFORMATION={counts.get('INFORMATION',0)}, TRADING={counts.get('TRADING',0)}")
    else:
        print(f"  {platform} PM: {len(df)} political videos")

    return df[["date"]].assign(source="prediction_markets", platform=platform)


def load_polls(json_path: Path, platform: str, date_col: str) -> pd.DataFrame:
    """Load raw poll-referencing video data."""
    if not json_path.exists():
        print(f"  {platform} Polls: NO DATA (run: python polls/scripts/polls_collection.py {platform.lower()})")
        return pd.DataFrame(columns=["date", "source", "platform"])

    data = json.load(open(json_path, encoding="utf-8"))
    df = pd.DataFrame(data)
    if date_col not in df.columns:
        print(f"  {platform} Polls: {len(data)} records but no '{date_col}' column")
        return pd.DataFrame(columns=["date", "source", "platform"])

    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"])
    print(f"  {platform} Polls (raw): {len(df)} videos")
    return df[["date"]].assign(source="polls", platform=platform)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def make_monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df.groupby(["month", "source", "platform"]).size().reset_index(name="count")


def compute_pm_share(counts: pd.DataFrame) -> pd.DataFrame:
    pivot = counts.pivot_table(
        index=["month", "platform"], columns="source", values="count", fill_value=0
    ).reset_index()
    for col in ["prediction_markets", "polls"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot["total"] = pivot["prediction_markets"] + pivot["polls"]
    pivot["pm_share"] = np.where(pivot["total"] > 0, pivot["prediction_markets"] / pivot["total"], np.nan)
    return pivot


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {"prediction_markets": "#E74C3C", "polls": "#3498DB"}
PLATFORM_STYLES = {"TikTok": "-", "YouTube": "--"}


def plot_figure(counts, pm_share, all_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Polls vs. Prediction Markets in Political Social Media",
                 fontsize=14, fontweight="bold", y=0.98)

    # Panel 1: TikTok monthly counts
    ax1 = axes[0, 0]
    tt = counts[counts["platform"] == "TikTok"]
    for src, color in COLORS.items():
        s = tt[tt["source"] == src]
        if not s.empty:
            ax1.plot(s["month"], s["count"], color=color, marker="o", markersize=3,
                     linewidth=1.5, label=src.replace("_", " ").title())
    ax1.set_title("TikTok: Monthly Video Counts", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Videos per month")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Panel 2: YouTube monthly counts
    ax2 = axes[0, 1]
    yt = counts[counts["platform"] == "YouTube"]
    for src, color in COLORS.items():
        s = yt[yt["source"] == src]
        if not s.empty:
            ax2.plot(s["month"], s["count"], color=color, marker="o", markersize=3,
                     linewidth=1.5, label=src.replace("_", " ").title())
    ax2.set_title("YouTube: Monthly Video Counts", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Videos per month")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Panel 3: PM share over time
    ax3 = axes[1, 0]
    for plat, style in PLATFORM_STYLES.items():
        s = pm_share[pm_share["platform"] == plat].dropna(subset=["pm_share"])
        if not s.empty:
            ax3.plot(s["month"], s["pm_share"] * 100, style, color="#2C3E50",
                     linewidth=1.5, marker="o", markersize=3, label=plat)
    ax3.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax3.set_title("Prediction Market Share of Citations", fontsize=11, fontweight="bold")
    ax3.set_ylabel("PM / (PM + Polls) %")
    ax3.set_ylim(-5, 105)
    ax3.legend(fontsize=9)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    # Panel 4: Cumulative growth
    ax4 = axes[1, 1]
    for plat in ["TikTok", "YouTube"]:
        for src, color in COLORS.items():
            s = all_df[(all_df["platform"] == plat) & (all_df["source"] == src)].sort_values("date")
            if s.empty:
                continue
            style = PLATFORM_STYLES[plat]
            label = f"{plat} — {src.replace('_', ' ').title()}"
            ax4.plot(s["date"], range(1, len(s) + 1), style, color=color,
                     linewidth=1.5, label=label, alpha=0.8)
    ax4.set_title("Cumulative Video Growth", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Total videos")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    outpath = OUT / "polls_vs_pm_temporal.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"\n  Figure saved: {outpath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(all_df, counts, pm_share):
    print("\n" + "=" * 60)
    print("POLLS vs. PREDICTION MARKETS: SUMMARY")
    print("=" * 60)

    for plat in ["TikTok", "YouTube"]:
        print(f"\n--- {plat} ---")
        for src in ["prediction_markets", "polls"]:
            n = len(all_df[(all_df["platform"] == plat) & (all_df["source"] == src)])
            print(f"  {src.replace('_', ' ').title():25s}: {n:,} videos")

    print("\n--- Date Ranges ---")
    for plat in ["TikTok", "YouTube"]:
        for src in ["prediction_markets", "polls"]:
            sub = all_df[(all_df["platform"] == plat) & (all_df["source"] == src)]
            if not sub.empty:
                print(f"  {plat} {src.replace('_', ' '):20s}: {sub['date'].min():%Y-%m-%d} to {sub['date'].max():%Y-%m-%d}")

    print("\n--- Peak Months ---")
    for plat in ["TikTok", "YouTube"]:
        for src in ["prediction_markets", "polls"]:
            sub = counts[(counts["platform"] == plat) & (counts["source"] == src)]
            if not sub.empty:
                peak = sub.loc[sub["count"].idxmax()]
                print(f"  {plat} {src.replace('_', ' '):20s}: {peak['month']:%Y-%m} ({peak['count']:,} videos)")

    print("\n--- PM Share by Year ---")
    yearly = all_df.copy()
    yearly["year"] = yearly["date"].dt.year
    for plat in ["TikTok", "YouTube"]:
        print(f"\n  {plat}:")
        plat_data = yearly[yearly["platform"] == plat]
        for year in sorted(plat_data["year"].unique()):
            yr = plat_data[plat_data["year"] == year]
            pm_n = len(yr[yr["source"] == "prediction_markets"])
            poll_n = len(yr[yr["source"] == "polls"])
            total = pm_n + poll_n
            if total > 0:
                print(f"    {year}: PM={pm_n:,}, Polls={poll_n:,}, PM share={pm_n/total*100:.1f}%")

    print("\n--- 2024 Election Quarter (Oct-Nov) ---")
    eq = all_df[(all_df["date"] >= "2024-10-01") & (all_df["date"] < "2024-12-01")]
    for plat in ["TikTok", "YouTube"]:
        p = eq[eq["platform"] == plat]
        pm_n = len(p[p["source"] == "prediction_markets"])
        poll_n = len(p[p["source"] == "polls"])
        total = pm_n + poll_n
        if total > 0:
            print(f"  {plat}: PM={pm_n:,}, Polls={poll_n:,}, PM share={pm_n/total*100:.1f}%")
        else:
            print(f"  {plat}: no data")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading datasets...")

    tt_pm = load_pm(TIKTOK_PM_CSV, TIKTOK_PM_INFO, "TikTok", "create_time")
    yt_pm = load_pm(YOUTUBE_PM_CSV, YOUTUBE_PM_INFO, "YouTube", "date_posted")
    tt_polls = load_polls(TIKTOK_POLLS_RAW, "TikTok", "create_time")
    yt_polls = load_polls(YOUTUBE_POLLS_RAW, "YouTube", "date_posted")

    dfs = [d for d in [tt_pm, yt_pm, tt_polls, yt_polls] if len(d) > 0]
    if not dfs:
        print("\nERROR: No data loaded.")
        sys.exit(1)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[(all_df["date"] >= "2020-01-01") & (all_df["date"] <= "2026-12-31")]

    has_polls = len(all_df[all_df["source"] == "polls"]) > 0
    if not has_polls:
        print("\n  NOTE: Poll data not yet collected.")
        print("  Run: python polls/scripts/polls_collection.py both")
        print("  Showing PM-only temporal analysis for now.\n")

    counts = make_monthly_counts(all_df)
    pm_share = compute_pm_share(counts)
    plot_figure(counts, pm_share, all_df)
    print_summary(all_df, counts, pm_share)


if __name__ == "__main__":
    main()
