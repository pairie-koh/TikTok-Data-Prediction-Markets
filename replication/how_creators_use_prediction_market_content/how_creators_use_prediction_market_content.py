"""
Replication script for: How creators use prediction market content

Generates a horizontal stacked bar chart showing the Information vs Trading
split for TikTok and YouTube prediction market videos.

Data sources:
  tiktok/data/info_vs_trading_progress.json
  youtube/data/info_vs_trading_progress.json

Output: how_creators_use_prediction_market_content.png (in this folder)

Usage:
    python replication/how_creators_use_prediction_market_content/how_creators_use_prediction_market_content.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = SCRIPT_DIR

TT_INFO = ROOT / "tiktok" / "data" / "info_vs_trading_progress.json"
YT_INFO = ROOT / "youtube" / "data" / "info_vs_trading_progress.json"


def main():
    if not TT_INFO.exists() or not YT_INFO.exists():
        print("ERROR: info_vs_trading_progress.json not found for one or both platforms.")
        sys.exit(1)

    tt_counts = Counter(json.load(open(TT_INFO, encoding="utf-8")).values())
    yt_counts = Counter(json.load(open(YT_INFO, encoding="utf-8")).values())

    tt_info = tt_counts.get("INFORMATION", 0)
    tt_trade = tt_counts.get("TRADING", 0)
    tt_total = tt_info + tt_trade

    yt_info = yt_counts.get("INFORMATION", 0)
    yt_trade = yt_counts.get("TRADING", 0)
    yt_total = yt_info + yt_trade

    print(f"TikTok:  INFORMATION={tt_info}, TRADING={tt_trade}, total={tt_total}")
    print(f"YouTube: INFORMATION={yt_info}, TRADING={yt_trade}, total={yt_total}")

    tt_info_pct = tt_info / tt_total * 100
    tt_trade_pct = tt_trade / tt_total * 100
    yt_info_pct = yt_info / yt_total * 100
    yt_trade_pct = yt_trade / yt_total * 100

    platforms = [f"TikTok\n(n={tt_total:,})", f"YouTube\n(n={yt_total:,})"]
    info_pcts = [tt_info_pct, yt_info_pct]
    trade_pcts = [tt_trade_pct, yt_trade_pct]
    info_counts = [tt_info, yt_info]
    trade_counts = [tt_trade, yt_trade]

    fig, ax = plt.subplots(figsize=(12, 5))

    bars_info = ax.barh(platforms, info_pcts, color="#2563EB", edgecolor="white", linewidth=0.5,
                        label="Information")
    bars_trade = ax.barh(platforms, trade_pcts, left=info_pcts, color="#F97316", edgecolor="white",
                         linewidth=0.5, label="Trading")

    # Labels on information bars
    for bar, pct, count in zip(bars_info, info_pcts, info_counts):
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%  ({count:,})", ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

    # Labels on trading bars
    for bar, pct, count, left in zip(bars_trade, trade_pcts, trade_counts, info_pcts):
        ax.text(left + pct / 2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%  ({count:,})", ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of videos (%)", fontsize=12)
    ax.set_title("How creators use prediction market content", fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")

    # Subtitle
    fig.text(0.5, 0.95,
             "On both platforms, the vast majority of prediction market videos cite odds as\n"
             "political evidence — not as trading advice. The market price has replaced the poll number.",
             ha="center", va="bottom", fontsize=10, style="italic", color="#555555")

    ax.tick_params(labelsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    outpath = OUTPUT_DIR / "how_creators_use_prediction_market_content.png"
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == "__main__":
    main()
