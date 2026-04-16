"""
Over-time comparison: Polls vs Prediction Markets citations
on TikTok and YouTube.

Produces:
  1. Monthly time series of video counts (polls vs PM, per platform)
  2. Monthly view counts
  3. Polls-share metric: polls / (polls + PM) over time
  4. Summary stats and CSV export

Usage:
    python polls/scripts/polls_vs_pm_comparison.py
    python polls/scripts/polls_vs_pm_comparison.py --plot    # also save charts
"""

import argparse
import csv
import io
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

csv.field_size_limit(10_000_000)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS = {
    "tiktok_pm":    ROOT / "tiktok" / "data" / "tiktok_platform_filtered.csv",
    "youtube_pm":   ROOT / "youtube" / "data" / "youtube_platform_filtered.csv",
    "tiktok_polls": ROOT / "polls"  / "data" / "tiktok_polls_filtered.csv",
    "youtube_polls": ROOT / "polls" / "data" / "youtube_polls_filtered.csv",
}
OUTPUT_DIR = ROOT / "polls" / "data"


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_yes_videos(path: Path, date_col: str, views_col: str) -> list[dict]:
    """Load YES-classified videos with parsed dates."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("_topic") != "YES":
                continue
            date_str = (r.get(date_col) or "")[:10]
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue
            views = 0
            try:
                views = int(r.get(views_col) or 0)
            except (ValueError, TypeError):
                pass
            rows.append({
                "date": dt,
                "month": dt.strftime("%Y-%m"),
                "views": views,
            })
    return rows


def load_all():
    data = {}
    configs = {
        "tiktok_pm":    ("create_time", "play_count"),
        "youtube_pm":   ("date_posted",  "views"),
        "tiktok_polls": ("create_time", "play_count"),
        "youtube_polls": ("date_posted",  "views"),
    }
    for key, (date_col, views_col) in configs.items():
        path = DATASETS[key]
        rows = load_yes_videos(path, date_col, views_col)
        data[key] = rows
        print(f"  {key}: {len(rows)} YES videos with valid dates")
    return data


# ---------------------------------------------------------------------------
# Aggregate by month
# ---------------------------------------------------------------------------

def monthly_agg(rows: list[dict]) -> dict[str, dict]:
    """Returns {month: {count, views}}"""
    agg = defaultdict(lambda: {"count": 0, "views": 0})
    for r in rows:
        m = r["month"]
        agg[m]["count"] += 1
        agg[m]["views"] += r["views"]
    return dict(agg)


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------

def build_comparison(data: dict) -> list[dict]:
    """Build month-by-month comparison across all 4 datasets."""
    aggs = {k: monthly_agg(v) for k, v in data.items()}

    # Get all months across all datasets
    all_months = set()
    for a in aggs.values():
        all_months.update(a.keys())
    all_months = sorted(all_months)

    # Filter to meaningful range (2020+ for TikTok overlap)
    all_months = [m for m in all_months if m >= "2020-01"]

    rows = []
    for m in all_months:
        tk_pm = aggs["tiktok_pm"].get(m, {"count": 0, "views": 0})
        tk_po = aggs["tiktok_polls"].get(m, {"count": 0, "views": 0})
        yt_pm = aggs["youtube_pm"].get(m, {"count": 0, "views": 0})
        yt_po = aggs["youtube_polls"].get(m, {"count": 0, "views": 0})

        # Polls share = polls / (polls + PM)
        tk_total = tk_pm["count"] + tk_po["count"]
        yt_total = yt_pm["count"] + yt_po["count"]

        tk_polls_share = tk_po["count"] / tk_total if tk_total > 0 else None
        yt_polls_share = yt_po["count"] / yt_total if yt_total > 0 else None

        tk_total_views = tk_pm["views"] + tk_po["views"]
        yt_total_views = yt_pm["views"] + yt_po["views"]
        tk_polls_view_share = tk_po["views"] / tk_total_views if tk_total_views > 0 else None
        yt_polls_view_share = yt_po["views"] / yt_total_views if yt_total_views > 0 else None

        rows.append({
            "month": m,
            "tk_pm_count": tk_pm["count"],
            "tk_polls_count": tk_po["count"],
            "tk_total": tk_total,
            "tk_polls_share": tk_polls_share,
            "tk_pm_views": tk_pm["views"],
            "tk_polls_views": tk_po["views"],
            "tk_polls_view_share": tk_polls_view_share,
            "yt_pm_count": yt_pm["count"],
            "yt_polls_count": yt_po["count"],
            "yt_total": yt_total,
            "yt_polls_share": yt_polls_share,
            "yt_pm_views": yt_pm["views"],
            "yt_polls_views": yt_po["views"],
            "yt_polls_view_share": yt_polls_view_share,
        })

    return rows


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_summary(data: dict, comparison: list[dict]):
    print("\n" + "=" * 70)
    print("POLLS vs PREDICTION MARKETS — OVER-TIME COMPARISON")
    print("=" * 70)

    # Overall counts
    print("\n--- Dataset sizes (YES videos with valid dates) ---")
    for key, rows in data.items():
        total_views = sum(r["views"] for r in rows)
        print(f"  {key:20s}: {len(rows):>5} videos, {total_views:>15,} views")

    # Annual breakdown
    print("\n--- Annual video counts ---")
    print(f"{'Year':>6}  {'TK PM':>6} {'TK Polls':>9} {'TK Share':>9}  {'YT PM':>6} {'YT Polls':>9} {'YT Share':>9}")
    print("-" * 70)

    yearly = defaultdict(lambda: {"tk_pm": 0, "tk_po": 0, "yt_pm": 0, "yt_po": 0})
    for r in comparison:
        y = r["month"][:4]
        yearly[y]["tk_pm"] += r["tk_pm_count"]
        yearly[y]["tk_po"] += r["tk_polls_count"]
        yearly[y]["yt_pm"] += r["yt_pm_count"]
        yearly[y]["yt_po"] += r["yt_polls_count"]

    for y in sorted(yearly.keys()):
        d = yearly[y]
        tk_t = d["tk_pm"] + d["tk_po"]
        yt_t = d["yt_pm"] + d["yt_po"]
        tk_sh = f"{d['tk_po']/tk_t*100:.1f}%" if tk_t > 0 else "—"
        yt_sh = f"{d['yt_po']/yt_t*100:.1f}%" if yt_t > 0 else "—"
        print(f"{y:>6}  {d['tk_pm']:>6} {d['tk_po']:>9} {tk_sh:>9}  {d['yt_pm']:>6} {d['yt_po']:>9} {yt_sh:>9}")

    # Key periods
    print("\n--- Key periods (monthly avg video count) ---")
    periods = [
        ("Pre-2024 election", "2023-01", "2024-06"),
        ("Campaign peak",     "2024-07", "2024-11"),
        ("Post-election",     "2024-12", "2025-06"),
        ("2025 H2+",          "2025-07", "2026-12"),
    ]

    for label, start, end in periods:
        period_rows = [r for r in comparison if start <= r["month"] <= end]
        if not period_rows:
            continue
        n = len(period_rows)
        tk_pm_avg = sum(r["tk_pm_count"] for r in period_rows) / n
        tk_po_avg = sum(r["tk_polls_count"] for r in period_rows) / n
        yt_pm_avg = sum(r["yt_pm_count"] for r in period_rows) / n
        yt_po_avg = sum(r["yt_polls_count"] for r in period_rows) / n

        tk_t = tk_pm_avg + tk_po_avg
        yt_t = yt_pm_avg + yt_po_avg
        tk_sh = f"{tk_po_avg/tk_t*100:.0f}%" if tk_t > 0 else "—"
        yt_sh = f"{yt_po_avg/yt_t*100:.0f}%" if yt_t > 0 else "—"

        print(f"  {label:25s}  TK: {tk_pm_avg:.1f} PM, {tk_po_avg:.1f} Polls ({tk_sh} polls)")
        print(f"  {'':25s}  YT: {yt_pm_avg:.1f} PM, {yt_po_avg:.1f} Polls ({yt_sh} polls)")

    # Platform comparison
    print("\n--- Platform comparison (overall) ---")
    tk_pm_total = sum(r["tk_pm_count"] for r in comparison)
    tk_po_total = sum(r["tk_polls_count"] for r in comparison)
    yt_pm_total = sum(r["yt_pm_count"] for r in comparison)
    yt_po_total = sum(r["yt_polls_count"] for r in comparison)

    tk_t = tk_pm_total + tk_po_total
    yt_t = yt_pm_total + yt_po_total
    print(f"  TikTok:  {tk_pm_total} PM + {tk_po_total} Polls = {tk_t} total")
    if tk_t:
        print(f"           Polls share: {tk_po_total/tk_t*100:.1f}%  |  PM share: {tk_pm_total/tk_t*100:.1f}%")
    print(f"  YouTube: {yt_pm_total} PM + {yt_po_total} Polls = {yt_t} total")
    if yt_t:
        print(f"           Polls share: {yt_po_total/yt_t*100:.1f}%  |  PM share: {yt_pm_total/yt_t*100:.1f}%")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(comparison: list[dict]):
    out = OUTPUT_DIR / "polls_vs_pm_monthly.csv"
    fieldnames = list(comparison[0].keys())
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in comparison:
            row = {**r}
            for k in ("tk_polls_share", "yt_polls_share", "tk_polls_view_share", "yt_polls_view_share"):
                if row[k] is not None:
                    row[k] = f"{row[k]:.4f}"
                else:
                    row[k] = ""
            w.writerow(row)
    print(f"\nSaved monthly comparison to {out}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(comparison: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    months = [datetime.strptime(r["month"], "%Y-%m") for r in comparison]

    # Filter to 2022+ for cleaner plots (TikTok PM didn't really exist before)
    idx = [i for i, m in enumerate(months) if m >= datetime(2022, 1, 1)]
    months = [months[i] for i in idx]
    comp = [comparison[i] for i in idx]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: TikTok video counts
    ax = axes[0]
    ax.bar(months, [r["tk_pm_count"] for r in comp], width=25, alpha=0.7, label="Prediction Markets", color="#2196F3")
    ax.bar(months, [r["tk_polls_count"] for r in comp], width=25, alpha=0.7, bottom=[r["tk_pm_count"] for r in comp], label="Polls", color="#FF9800")
    ax.set_ylabel("Video count")
    ax.set_title("TikTok: Polls vs Prediction Markets (monthly video count)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: YouTube video counts
    ax = axes[1]
    ax.bar(months, [r["yt_pm_count"] for r in comp], width=25, alpha=0.7, label="Prediction Markets", color="#2196F3")
    ax.bar(months, [r["yt_polls_count"] for r in comp], width=25, alpha=0.7, bottom=[r["yt_pm_count"] for r in comp], label="Polls", color="#FF9800")
    ax.set_ylabel("Video count")
    ax.set_title("YouTube: Polls vs Prediction Markets (monthly video count)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Polls share (% of total that cites polls)
    ax = axes[2]
    tk_share = [r["tk_polls_share"] for r in comp]
    yt_share = [r["yt_polls_share"] for r in comp]
    # Plot only where data exists
    tk_m = [months[i] for i in range(len(comp)) if tk_share[i] is not None]
    tk_s = [s for s in tk_share if s is not None]
    yt_m = [months[i] for i in range(len(comp)) if yt_share[i] is not None]
    yt_s = [s for s in yt_share if s is not None]

    ax.plot(tk_m, [s * 100 for s in tk_s], marker="o", markersize=3, label="TikTok", color="#E91E63", linewidth=1.5)
    ax.plot(yt_m, [s * 100 for s in yt_s], marker="s", markersize=3, label="YouTube", color="#9C27B0", linewidth=1.5)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="50% line")
    ax.set_ylabel("Polls share (%)")
    ax.set_xlabel("Month")
    ax.set_title("Polls share: % of political discussion citing polls (vs prediction markets)")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "polls_vs_pm_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Generate charts")
    args = parser.parse_args()

    print("Loading datasets...")
    data = load_all()

    print("\nBuilding monthly comparison...")
    comparison = build_comparison(data)

    print_summary(data, comparison)
    save_csv(comparison)

    if args.plot:
        print("\nGenerating plots...")
        make_plots(comparison)


if __name__ == "__main__":
    main()
