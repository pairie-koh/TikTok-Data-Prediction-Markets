"""
Extract specific contracts, races, and candidates mentioned in YouTube prediction market videos.

Same extraction as tiktok/scripts/extract_contracts.py but for YouTube data.

Usage:
    export OPENROUTER_API_KEY=...
    python youtube/scripts/extract_contracts.py
"""

import csv
import io
import json
import os
import sys
import time
from pathlib import Path

import requests

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

csv.field_size_limit(10_000_000)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILTERED_CSV = DATA_DIR / "youtube_platform_filtered.csv"
PROGRESS_FILE = DATA_DIR / "contract_extraction_progress.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "anthropic/claude-sonnet-4"

EXTRACT_PROMPT = """You are extracting structured data about prediction market contracts and political races mentioned in a YouTube video.

Given the video's title, description, and transcript, extract:

1. race_types: Which types of races/contracts are discussed? Pick ALL that apply from:
   - "presidential" (president, White House)
   - "senate" (US Senate races)
   - "house" (US House/Congressional races)
   - "governor" (gubernatorial races)
   - "policy" (non-election: tariffs, Fed rates, regulation, crypto, etc.)
   - "party_control" (which party controls House/Senate/Congress overall)
   - "other" (anything else)

2. race_specifics: List each specific race or contract mentioned. Use format like:
   - "2024 Presidential"
   - "2026 GA Senate"
   - "2026 NM-2 House"
   - "2026 House Control"
   - "Fed Rate Cut 2025"
   Be specific about year and location.

3. candidates_mentioned: List every political figure or candidate mentioned by name.
   Use their common name (e.g., "Trump", "Harris", "DeSantis", "AOC").

4. contracts_cited: List specific prediction market contracts or bets described.
   E.g., "Trump wins 2024 election", "Democrats flip the House", "Fed cuts rates in March"

5. odds_mentioned: List any specific probabilities or odds cited.
   Format as: "64% - Democrats flip the House" or "55% - Trump wins"

Respond with ONLY valid JSON, no markdown fences:
{{"race_types": [...], "race_specifics": [...], "candidates_mentioned": [...], "contracts_cited": [...], "odds_mentioned": [...]}}

If a field has no matches, use an empty list [].

---
Title: {title}
Description: {description}
Transcript: {transcript}"""


def load_politics_videos() -> list[dict]:
    """Load only _topic=YES videos from the filtered CSV."""
    videos = []
    with open(FILTERED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("_topic", "").upper() == "YES":
                videos.append(row)
    return videos


def llm_extract(title: str, description: str, transcript: str) -> dict:
    """Call OpenRouter API to extract contract data."""
    prompt = EXTRACT_PROMPT.format(
        title=title[:200],
        description=description[:500],
        transcript=transcript[:2000],
    )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    return json.loads(text)


def main():
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY environment variable.")
        sys.exit(1)

    print("Loading YouTube US politics videos (_topic=YES)...")
    videos = load_politics_videos()
    print(f"  Loaded {len(videos)} videos")

    # Load existing progress
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"  Resuming: {len(progress)} already extracted")

    remaining = [v for v in videos if v.get("video_id", "") not in progress]
    print(f"  Remaining: {len(remaining)}\n")

    api_calls = 0
    errors = 0

    for i, v in enumerate(remaining):
        vid = v.get("video_id", "")
        title = v.get("title", "") or ""
        description = v.get("description", "") or ""
        transcript = v.get("_transcript_text", "") or ""

        if not transcript and not description and not title:
            progress[vid] = {
                "race_types": [],
                "race_specifics": [],
                "candidates_mentioned": [],
                "contracts_cited": [],
                "odds_mentioned": [],
            }
            continue

        try:
            result = llm_extract(title, description, transcript)

            for key in ["race_types", "race_specifics", "candidates_mentioned", "contracts_cited", "odds_mentioned"]:
                if key not in result or not isinstance(result[key], list):
                    result[key] = []

            progress[vid] = result
            api_calls += 1

        except Exception as e:
            print(f"  Error on {vid}: {e}")
            progress[vid] = {
                "race_types": ["error"],
                "race_specifics": [],
                "candidates_mentioned": [],
                "contracts_cited": [],
                "odds_mentioned": [],
                "_error": str(e),
            }
            errors += 1
            time.sleep(2)

        # Save progress every 50
        if (api_calls + errors) % 50 == 0 and (api_calls + errors) > 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            print(f"  [{i+1}/{len(remaining)}] API calls: {api_calls}, errors: {errors}")

        time.sleep(0.3)

    # Final save
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    print(f"\nDone. API calls: {api_calls}, errors: {errors}")
    print(f"Total extracted: {len(progress)}")

    from collections import Counter
    race_type_counter = Counter()
    candidate_counter = Counter()
    for data in progress.values():
        for rt in data.get("race_types", []):
            race_type_counter[rt] += 1
        for c in data.get("candidates_mentioned", []):
            candidate_counter[c] += 1

    print(f"\nRace type distribution:")
    for rt, count in race_type_counter.most_common():
        print(f"  {rt}: {count}")

    print(f"\nTop 15 candidates mentioned:")
    for c, count in candidate_counter.most_common(15):
        print(f"  {c}: {count}")


if __name__ == "__main__":
    main()
