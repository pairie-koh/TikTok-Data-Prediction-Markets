"""
Two LLM classifications in one pass (to save API calls):

1. FRAMING: Does the creator present prediction market odds as neutral data
   or as partisan ammunition?

2. STALENESS: Does the video cite specific odds/probabilities that would
   become outdated, or is it general commentary?

Usage:
    set OPENROUTER_API_KEY=...
    python scripts/classify_framing_and_staleness.py

    python scripts/classify_framing_and_staleness.py --dry-run
"""

import argparse
import csv
import io
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILTERED_CSV = DATA_DIR / "tiktok_platform_filtered.csv"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
RELEVANCE_FILE = DATA_DIR / "relevance_progress.json"
INFO_TRADING_FILE = DATA_DIR / "info_vs_trading_progress.json"
PROGRESS_FILE = DATA_DIR / "framing_staleness_progress.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

CLASSIFY_PROMPT = """Analyze this TikTok video about prediction markets and US politics. Answer TWO questions.

QUESTION 1 — FRAMING: How does the creator present the prediction market odds?

NEUTRAL — The creator presents odds as objective data or interesting information without pushing a political position. They report what the market says without arguing it proves their side is right.
Examples: "Here's what Polymarket shows for the midterms", "Odds shifted 5 points after the debate"

PARTISAN — The creator uses prediction market odds as ammunition to support their political position. They cite odds to argue their candidate is winning, the other side is losing, or to validate their political take.
Examples: "See? Even the betting markets know Trump is going to win", "Polymarket proves the Democrats are finished"

QUESTION 2 — STALENESS: Does this video cite specific odds or probabilities that would become outdated?

CITES_ODDS — The video mentions specific probability numbers, percentages, or price movements that are tied to a moment in time and would be stale if watched later.
Examples: "Trump is at 62% right now", "Odds dropped from 55 to 48 this week", "Kalshi has the GOP at 73%"

NO_ODDS — The video discusses prediction markets generally without citing specific numbers that would expire. The content would still make sense watched weeks later.
Examples: "Prediction markets are more accurate than polls", "You can bet on elections on Polymarket", "Here's how prediction market odds work"

Respond with ONLY valid JSON on a single line:
{{"framing": "NEUTRAL" or "PARTISAN", "staleness": "CITES_ODDS" or "NO_ODDS"}}

---
Description: {description}
Transcript: {transcript}"""


def load_relevant_politics_videos():
    relevance = {}
    if RELEVANCE_FILE.exists():
        with open(RELEVANCE_FILE, encoding="utf-8") as f:
            relevance = json.load(f)

    videos = []
    with open(FILTERED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row.get("post_id", "")
            if row.get("_topic", "").upper() == "YES" and relevance.get(pid, "") == "RELEVANT":
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


def classify(client, description, transcript):
    prompt = CLASSIFY_PROMPT.format(
        description=description[:500],
        transcript=transcript[:1500],
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=50,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    result = json.loads(text)

    framing = result.get("framing", "UNKNOWN").upper()
    staleness = result.get("staleness", "UNKNOWN").upper()

    if framing not in ("NEUTRAL", "PARTISAN"):
        framing = "UNKNOWN"
    if staleness not in ("CITES_ODDS", "NO_ODDS"):
        staleness = "UNKNOWN"

    return {"framing": framing, "staleness": staleness}


def safe_int(val, default=0):
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    videos = load_relevant_politics_videos()
    print(f"Loaded {len(videos)} relevant US politics videos\n")

    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"Already classified: {len(progress)}")

    remaining = [v for v in videos if v.get("post_id", "") not in progress]
    print(f"Remaining: {len(remaining)}\n")

    if args.dry_run:
        print(f"[DRY RUN] Would classify {len(remaining)} videos")
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

        try:
            result = classify(client, description, transcript)
            progress[pid] = result
            api_calls += 1
        except Exception as e:
            print(f"  Error on {pid}: {e}")
            progress[pid] = {"framing": "ERROR", "staleness": "ERROR"}
            errors += 1

        if (api_calls + errors) % 25 == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f)
            neut = sum(1 for v in progress.values() if v["framing"] == "NEUTRAL")
            part = sum(1 for v in progress.values() if v["framing"] == "PARTISAN")
            odds = sum(1 for v in progress.values() if v["staleness"] == "CITES_ODDS")
            print(f"  [{i+1}/{len(remaining)}] calls: {api_calls} | NEUTRAL: {neut}, PARTISAN: {part} | CITES_ODDS: {odds}")

        time.sleep(0.15)

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    # Load content_type for cross-tabs
    content_types = {}
    if INFO_TRADING_FILE.exists():
        with open(INFO_TRADING_FILE, encoding="utf-8") as f:
            content_types = json.load(f)

    # ── Results ──
    print(f"\n{'=' * 60}")
    print(f"  RESULTS ({len(progress)} videos)")
    print(f"{'=' * 60}\n")

    print(f"API calls: {api_calls}, errors: {errors}\n")

    n = len(progress)

    # Framing distribution
    framing_counts = Counter(v["framing"] for v in progress.values())
    print(f"FRAMING:")
    print(f"  {'Category':<15} {'Count':>6} {'%':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*7}")
    for cat in ["NEUTRAL", "PARTISAN", "UNKNOWN", "ERROR"]:
        c = framing_counts.get(cat, 0)
        if c > 0:
            print(f"  {cat:<15} {c:>6} {c/n*100:>6.1f}%")

    # Staleness distribution
    stale_counts = Counter(v["staleness"] for v in progress.values())
    print(f"\nSTALENESS:")
    print(f"  {'Category':<15} {'Count':>6} {'%':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*7}")
    for cat in ["CITES_ODDS", "NO_ODDS", "UNKNOWN", "ERROR"]:
        c = stale_counts.get(cat, 0)
        if c > 0:
            print(f"  {cat:<15} {c:>6} {c/n*100:>6.1f}%")

    # Cross-tab: framing x staleness
    print(f"\nFRAMING x STALENESS:")
    cross = Counter()
    for v in progress.values():
        cross[(v["framing"], v["staleness"])] += 1
    print(f"  {'':15} {'CITES_ODDS':>12} {'NO_ODDS':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for fr in ["NEUTRAL", "PARTISAN"]:
        print(f"  {fr:<15} {cross.get((fr,'CITES_ODDS'),0):>12} {cross.get((fr,'NO_ODDS'),0):>12}")

    # Cross-tab: content_type x framing
    print(f"\nCONTENT TYPE x FRAMING:")
    cross2 = Counter()
    for pid, data in progress.items():
        ct = content_types.get(pid, "UNKNOWN")
        cross2[(ct, data["framing"])] += 1
    print(f"  {'':15} {'NEUTRAL':>12} {'PARTISAN':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for ct in ["INFORMATION", "TRADING"]:
        print(f"  {ct:<15} {cross2.get((ct,'NEUTRAL'),0):>12} {cross2.get((ct,'PARTISAN'),0):>12}")

    # Cross-tab: content_type x staleness
    print(f"\nCONTENT TYPE x STALENESS:")
    cross3 = Counter()
    for pid, data in progress.items():
        ct = content_types.get(pid, "UNKNOWN")
        cross3[(ct, data["staleness"])] += 1
    print(f"  {'':15} {'CITES_ODDS':>12} {'NO_ODDS':>12}")
    print(f"  {'-'*15} {'-'*12} {'-'*12}")
    for ct in ["INFORMATION", "TRADING"]:
        print(f"  {ct:<15} {cross3.get((ct,'CITES_ODDS'),0):>12} {cross3.get((ct,'NO_ODDS'),0):>12}")

    # Engagement by framing
    print(f"\nENGAGEMENT BY FRAMING:")
    for fr in ["NEUTRAL", "PARTISAN"]:
        fr_vids = [v for v in videos if progress.get(v.get("post_id",""), {}).get("framing") == fr]
        if not fr_vids:
            continue
        views = [safe_int(v.get("play_count")) for v in fr_vids]
        print(f"  {fr}: {len(fr_vids)} videos, mean views: {statistics.mean(views):,.0f}, median: {statistics.median(views):,.0f}")

    # Engagement by staleness
    print(f"\nENGAGEMENT BY STALENESS:")
    for st in ["CITES_ODDS", "NO_ODDS"]:
        st_vids = [v for v in videos if progress.get(v.get("post_id",""), {}).get("staleness") == st]
        if not st_vids:
            continue
        views = [safe_int(v.get("play_count")) for v in st_vids]
        print(f"  {st}: {len(st_vids)} videos, mean views: {statistics.mean(views):,.0f}, median: {statistics.median(views):,.0f}")


if __name__ == "__main__":
    main()
