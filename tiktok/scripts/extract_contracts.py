"""
Extract specific contracts, races, and candidates mentioned in TikTok prediction market videos.

For each video, extracts:
  - race_types: list of race types (presidential, senate, house, governor, policy, other)
  - race_specifics: list of specific races (e.g., "2024 Presidential", "GA Senate 2026")
  - candidates_mentioned: list of candidate names
  - contracts_cited: list of contract descriptions (e.g., "Trump wins presidency")
  - odds_mentioned: list of specific probabilities cited (e.g., "64%")

Uses OpenRouter API (Claude Sonnet).
Saves progress incrementally to allow resuming.

Usage:
    export OPENROUTER_API_KEY=...
    python tiktok/scripts/extract_contracts.py
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILTERED_CSV = DATA_DIR / "tiktok_platform_filtered.csv"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
PROGRESS_FILE = DATA_DIR / "contract_extraction_progress.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = "anthropic/claude-sonnet-4"

EXTRACT_PROMPT = """You are extracting structured data about prediction market contracts and political races mentioned in a TikTok video.

Given the video's description and transcript, extract:

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
   - "Trump Tariffs"
   Be specific. If they mention "the Georgia Senate race" that's "2026 GA Senate" (or 2024 depending on context).

3. candidates_mentioned: List every political figure or candidate mentioned by name.
   Use their common name (e.g., "Trump", "Harris", "DeSantis", "AOC").

4. contracts_cited: List specific prediction market contracts or bets described.
   E.g., "Trump wins 2024 election", "Democrats flip the House", "Bitcoin hits $100k",
   "Fed cuts rates in March", "Trump wins popular vote"

5. odds_mentioned: List any specific probabilities or odds cited.
   E.g., "64%", "55%", "$0.60" (for contract prices). Include the context briefly.
   Format as: "64% - Democrats flip the House" or "55% - Trump wins"

Respond with ONLY valid JSON, no markdown fences:
{{"race_types": [...], "race_specifics": [...], "candidates_mentioned": [...], "contracts_cited": [...], "odds_mentioned": [...]}}

If a field has no matches, use an empty list [].

---
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


def get_transcript(post_id: str, csv_transcript: str = "") -> str:
    """Load transcript: whisper > tiktok captions > csv fallback."""
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


def llm_extract(description: str, transcript: str) -> dict:
    """Call OpenRouter API to extract contract data."""
    prompt = EXTRACT_PROMPT.format(
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

    print("Loading US politics videos (_topic=YES)...")
    videos = load_politics_videos()
    print(f"  Loaded {len(videos)} videos")

    # Load existing progress
    progress = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"  Resuming: {len(progress)} already extracted")

    remaining = [v for v in videos if v.get("post_id", "") not in progress]
    print(f"  Remaining: {len(remaining)}\n")

    api_calls = 0
    errors = 0

    for i, v in enumerate(remaining):
        pid = v.get("post_id", "")
        description = v.get("description", "") or ""
        transcript = get_transcript(pid, v.get("_transcript", ""))

        if not transcript and not description:
            progress[pid] = {
                "race_types": [],
                "race_specifics": [],
                "candidates_mentioned": [],
                "contracts_cited": [],
                "odds_mentioned": [],
            }
            continue

        try:
            result = llm_extract(description, transcript)

            # Validate structure
            for key in ["race_types", "race_specifics", "candidates_mentioned", "contracts_cited", "odds_mentioned"]:
                if key not in result or not isinstance(result[key], list):
                    result[key] = []

            progress[pid] = result
            api_calls += 1

        except Exception as e:
            print(f"  Error on {pid}: {e}")
            progress[pid] = {
                "race_types": ["error"],
                "race_specifics": [],
                "candidates_mentioned": [],
                "contracts_cited": [],
                "odds_mentioned": [],
                "_error": str(e),
            }
            errors += 1
            time.sleep(2)

        # Save progress every 25
        if (api_calls + errors) % 25 == 0 and (api_calls + errors) > 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            print(f"  [{i+1}/{len(remaining)}] API calls: {api_calls}, errors: {errors}")

        time.sleep(0.3)

    # Final save
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    print(f"\nDone. API calls: {api_calls}, errors: {errors}")
    print(f"Total extracted: {len(progress)}")

    # Quick summary
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
