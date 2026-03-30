"""
Classify TikTok prediction market videos as INFORMATION vs TRADING.

INFORMATION: The video is about politics; prediction market odds are cited
             as evidence or context.
TRADING:     The video is about making money on prediction markets; the
             political event is incidental.

Usage:
    set OPENROUTER_API_KEY=...
    python scripts/classify_info_vs_trading.py

    # Preview without API calls:
    python scripts/classify_info_vs_trading.py --dry-run
"""

import argparse
import csv
import io
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILTERED_CSV = DATA_DIR / "tiktok_platform_filtered.csv"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
PROGRESS_FILE = DATA_DIR / "info_vs_trading_progress.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

CLASSIFY_PROMPT = """A TikTok video mentions a prediction market platform ({platforms}). Classify the PRIMARY PURPOSE of this video.

INFORMATION — The video is about politics. The creator discusses political events, candidates, elections, or policy, and references prediction market odds as a piece of evidence or context. The point of the video is the political topic, not making money.
Examples:
- "Trump's odds on Polymarket dropped after the debate"
- "Here's what prediction markets say about the midterms"
- "Kalshi has Newsom at 20% for 2028 — here's why that matters"

TRADING — The video is about making money on prediction markets. The creator focuses on trading strategies, profits, portfolio advice, or how to use the platform. The political event is incidental — it's just the thing being traded.
Examples:
- "I made $5K on Polymarket this month, here's how"
- "Best strategy for trading election contracts on Kalshi"
- "How to sign up for Polymarket and start trading"

Respond with exactly one word: INFORMATION or TRADING

---
Description: {description}
Transcript: {transcript}"""


def load_politics_videos():
    videos = []
    with open(FILTERED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("_topic", "").upper() == "YES":
                videos.append(row)
    return videos


def get_transcript(post_id, csv_transcript=""):
    whisper_file = WHISPER_DIR / f"{post_id}.txt"
    if whisper_file.exists():
        text = whisper_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    tiktok_file = TIKTOK_DIR / f"{post_id}.txt"
    if tiktok_file.exists():
        text = tiktok_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    return csv_transcript


def get_client():
    from openai import OpenAI
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY")
        sys.exit(1)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def classify(client, platforms, description, transcript):
    prompt = CLASSIFY_PROMPT.format(
        platforms=platforms,
        description=description[:500],
        transcript=transcript[:1500],
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip().upper()
    if text in ("INFORMATION", "TRADING"):
        return text
    if "INFORMATION" in text:
        return "INFORMATION"
    if "TRADING" in text:
        return "TRADING"
    return "UNKNOWN"


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Show what would be classified, no API calls")
    args = parser.parse_args()

    videos = load_politics_videos()
    print(f"Loaded {len(videos)} US politics videos\n")

    # Load progress
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"Already classified: {len(progress)}")

    remaining = [v for v in videos if v.get("post_id", "") not in progress]
    print(f"Remaining: {len(remaining)}\n")

    if args.dry_run:
        print("[DRY RUN] Would classify these videos:")
        for v in remaining[:5]:
            desc = (v.get("description", "") or "")[:80].replace("\n", " ")
            print(f"  {v.get('post_id', '')}: {desc}...")
        if len(remaining) > 5:
            print(f"  ... and {len(remaining) - 5} more")
        print(f"\nEstimated API calls: {len(remaining)}")
        return

    client = get_client()
    print(f"Model: {LLM_MODEL}")
    print(f"Classifying {len(remaining)} videos...\n")

    api_calls = 0
    errors = 0

    for i, v in enumerate(remaining):
        pid = v.get("post_id", "")
        description = v.get("description", "") or ""
        transcript = get_transcript(pid, v.get("_transcript", ""))
        platforms = v.get("_platforms", "")

        try:
            result = classify(client, platforms, description, transcript)
            progress[pid] = result
            api_calls += 1
        except Exception as e:
            print(f"  Error on {pid}: {e}")
            progress[pid] = "ERROR"
            errors += 1

        # Save every 25
        if (api_calls + errors) % 25 == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f)
            info_n = sum(1 for v in progress.values() if v == "INFORMATION")
            trade_n = sum(1 for v in progress.values() if v == "TRADING")
            print(f"  [{i+1}/{len(remaining)}] calls: {api_calls} | INFO: {info_n}, TRADING: {trade_n}")

        time.sleep(0.15)

    # Final save
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    # ── Results ──
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}\n")

    print(f"API calls: {api_calls}, errors: {errors}\n")

    # Overall distribution
    counts = Counter(progress.values())
    n = len(progress)
    print(f"Overall classification ({n} videos):")
    print(f"  {'Category':<15} {'Count':>6} {'%':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*7}")
    for cat in ["INFORMATION", "TRADING", "UNKNOWN", "ERROR"]:
        c = counts.get(cat, 0)
        if c > 0:
            print(f"  {cat:<15} {c:>6} {c/n*100:>6.1f}%")

    # Cross-tab by platform
    print(f"\nBy platform:")
    platform_data = {}
    for v in videos:
        pid = v.get("post_id", "")
        if pid not in progress:
            continue
        label = progress[pid]
        for p in v.get("_platforms", "").split(", "):
            p = p.strip()
            if p:
                if p not in platform_data:
                    platform_data[p] = Counter()
                platform_data[p][label] += 1

    for plat, cnts in sorted(platform_data.items()):
        total = sum(cnts.values())
        info = cnts.get("INFORMATION", 0)
        trade = cnts.get("TRADING", 0)
        print(f"  {plat:<15} INFO: {info:>4} ({info/total*100:.0f}%)  TRADING: {trade:>4} ({trade/total*100:.0f}%)  total: {total}")

    # Engagement comparison
    print(f"\nEngagement by category:")
    for cat in ["INFORMATION", "TRADING"]:
        cat_vids = [v for v in videos if progress.get(v.get("post_id", "")) == cat]
        if not cat_vids:
            continue
        views = [safe_int(v.get("play_count")) for v in cat_vids]
        likes = [safe_int(v.get("digg_count")) for v in cat_vids]
        import statistics
        print(f"  {cat}:")
        print(f"    Videos:      {len(cat_vids)}")
        print(f"    Total views: {sum(views):,}")
        print(f"    Mean views:  {statistics.mean(views):,.0f}")
        print(f"    Median views:{statistics.median(views):,.0f}")
        print(f"    Mean likes:  {statistics.mean(likes):,.0f}")


if __name__ == "__main__":
    main()
