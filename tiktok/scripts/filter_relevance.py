"""
Relevance filter: does the video genuinely discuss the matched prediction
market platform, or is it just hashtag spam / an ad for something else?

Usage:
    set OPENROUTER_API_KEY=...
    python scripts/filter_relevance.py

    # Preview without API calls:
    python scripts/filter_relevance.py --dry-run
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

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILTERED_CSV = DATA_DIR / "tiktok_platform_filtered.csv"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
PROGRESS_FILE = DATA_DIR / "relevance_progress.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

RELEVANCE_PROMPT = """A TikTok video matched our filter because it contains the keyword "{platforms}" in its {match_source}. But we need to verify: does this video genuinely discuss or reference {platforms} as a prediction market, or is the mention incidental?

RELEVANT — The video genuinely discusses {platforms}. This includes:
- Citing odds or prices from {platforms}
- Discussing events or markets on {platforms}
- Reviewing or commenting on {platforms} as a platform
- Showing {platforms} screenshots or data

NOT_RELEVANT — The mention of {platforms} is incidental. This includes:
- Hashtag spam: using #{platforms} for visibility but the video is about something else
- Ads for a different platform that just tag {platforms} as a competitor keyword
- Passing mention with no substantive discussion of {platforms}
- The video is about a completely different topic

Respond with exactly one word: RELEVANT or NOT_RELEVANT

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


def classify(client, platforms, match_source, description, transcript):
    prompt = RELEVANCE_PROMPT.format(
        platforms=platforms,
        match_source=match_source,
        description=description[:500],
        transcript=transcript[:1500],
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip().upper()
    if "NOT_RELEVANT" in text or "NOT RELEVANT" in text:
        return "NOT_RELEVANT"
    if "RELEVANT" in text:
        return "RELEVANT"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    videos = load_politics_videos()
    print(f"Loaded {len(videos)} US politics videos\n")

    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"Already classified: {len(progress)}")

    remaining = [v for v in videos if v.get("post_id", "") not in progress]
    print(f"Remaining: {len(remaining)}\n")

    if args.dry_run:
        print("[DRY RUN] Would check relevance for these videos:")
        for v in remaining[:5]:
            desc = (v.get("description", "") or "")[:80].replace("\n", " ")
            print(f"  {v.get('post_id','')}: [{v.get('_platforms','')}] {desc}...")
        if len(remaining) > 5:
            print(f"  ... and {len(remaining) - 5} more")
        print(f"\nEstimated API calls: {len(remaining)}")
        return

    client = get_client()
    print(f"Model: {LLM_MODEL}")
    print(f"Checking relevance for {len(remaining)} videos...\n")

    api_calls = 0
    errors = 0

    for i, v in enumerate(remaining):
        pid = v.get("post_id", "")
        description = v.get("description", "") or ""
        transcript = get_transcript(pid, v.get("_transcript", ""))
        platforms = v.get("_platforms", "")
        match_source = v.get("_match_source", "")

        try:
            result = classify(client, platforms, match_source, description, transcript)
            progress[pid] = result
            api_calls += 1
        except Exception as e:
            print(f"  Error on {pid}: {e}")
            progress[pid] = "ERROR"
            errors += 1

        if (api_calls + errors) % 25 == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f)
            rel = sum(1 for v in progress.values() if v == "RELEVANT")
            notrel = sum(1 for v in progress.values() if v == "NOT_RELEVANT")
            print(f"  [{i+1}/{len(remaining)}] calls: {api_calls} | RELEVANT: {rel}, NOT_RELEVANT: {notrel}")

        time.sleep(0.15)

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    # Results
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}\n")

    print(f"API calls: {api_calls}, errors: {errors}\n")

    counts = Counter(progress.values())
    n = len(progress)
    print(f"Relevance classification ({n} videos):")
    print(f"  {'Category':<15} {'Count':>6} {'%':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*7}")
    for cat in ["RELEVANT", "NOT_RELEVANT", "UNKNOWN", "ERROR"]:
        c = counts.get(cat, 0)
        if c > 0:
            print(f"  {cat:<15} {c:>6} {c/n*100:>6.1f}%")

    # Show NOT_RELEVANT videos
    not_relevant = [v for v in videos if progress.get(v.get("post_id", "")) == "NOT_RELEVANT"]
    if not_relevant:
        print(f"\nNOT_RELEVANT videos ({len(not_relevant)}):")
        for v in not_relevant:
            desc = (v.get("description", "") or "")[:100].replace("\n", " ")
            print(f"  {v.get('post_id','')}: [{v.get('_platforms','')}] {desc}")


if __name__ == "__main__":
    main()
