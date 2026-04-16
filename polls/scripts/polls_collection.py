"""
Collect poll-referencing videos from TikTok and YouTube via Bright Data.

Parallel dataset to the prediction market collections in tiktok/ and youtube/.
Same Bright Data API, same approach, different keywords.

Purpose: enable an over-time comparison of how often people cite
polls vs. prediction markets when discussing politics on each platform.

Usage:
    set BRIGHTDATA_API_TOKEN=...
    python polls/scripts/polls_collection.py tiktok
    python polls/scripts/polls_collection.py youtube
    python polls/scripts/polls_collection.py both
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
POLLS_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = POLLS_DIR / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

TIKTOK_RAW = DATA_RAW / "tiktok_polls_raw.json"
YOUTUBE_RAW = DATA_RAW / "youtube_polls_raw.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BRIGHTDATA_API_TOKEN = os.environ.get("BRIGHTDATA_API_TOKEN", "")
TIKTOK_DATASET_ID = "gd_lu702nij2f790tmv9h"   # TikTok Posts (same as PM pipeline)
YOUTUBE_DATASET_ID = "gd_lk56epmy2i5g7lzu0k"  # YouTube Posts (same as PM pipeline)

# ---------------------------------------------------------------------------
# Poll keywords — structured to mirror the PM keyword categories
#
# The PM pipelines use ~50-70 keywords pairing platform names (Polymarket,
# Kalshi) with political context. For polls, we pair polling sources/terms
# with political context in the same way.
# ---------------------------------------------------------------------------

# Category 1: Specific polling sources + election (mirrors PLATFORM_POLITICAL)
SOURCE_POLITICAL = [
    "538 election",
    "fivethirtyeight election",
    "fivethirtyeight forecast",
    "realclearpolitics polls",
    "realclearpolitics average",
    "nate silver polls",
    "nate silver forecast",
    "quinnipiac poll",
    "fox news poll",
    "cnn poll election",
    "nbc poll election",
    "new york times poll",
    "marist poll",
    "monmouth poll",
    "emerson poll",
    "trafalgar poll",
    "rasmussen poll",
    "morning consult poll",
]

# Category 2: Generic polls + election (mirrors GENERIC_POLITICAL)
GENERIC_POLITICAL = [
    "election polls",
    "election polls 2024",
    "election polling data",
    "polling average election",
    "poll results election",
    "latest election polls",
    "poll numbers election",
    "who is winning in the polls",
    "poll tracker election",
    "swing state polls",
    "battleground state polls",
    "national polls election",
    "polls show",
]

# Category 3: Polls + candidates (mirrors CANDIDATE_ODDS)
CANDIDATE_POLLS = [
    "trump polls",
    "harris polls",
    "biden polls",
    "trump poll numbers",
    "harris poll numbers",
    "trump vs harris polls",
    "trump vs biden polls",
    "desantis polls",
    "vance polls",
]

# Category 4: Event-specific (mirrors EVENT_SPECIFIC)
EVENT_SPECIFIC = [
    "debate polls",
    "post debate polls",
    "election night polls",
    "exit polls 2024",
]

# Category 5: Time-specific (mirrors TIME_SPECIFIC)
TIME_SPECIFIC = [
    "polls 2024",
    "polls 2026",
    "midterm polls 2026",
]

# Category 6: Key races (mirrors MIDTERM_RACES)
RACE_POLLS = [
    "georgia senate polls",
    "michigan senate polls",
    "north carolina senate polls",
    "ohio senate polls",
    "pennsylvania senate polls",
    "arizona senate polls",
]

# Category 7: Hashtags (mirrors HASHTAG_SEARCHES)
HASHTAG_SEARCHES = [
    "#electionpolls",
    "#polls2024",
    "#pollingdata",
    "#electionpolls2024",
    "#polltracker",
    "#electionforecast",
    "#pollresults",
]

# Category 8: Methodology / meta-polling (unique to polls)
METHODOLOGY = [
    "poll margin of error",
    "polls accuracy",
    "polls wrong 2024",
    "polls reliable",
    "poll sample size",
    "likely voter polls",
]

# Category 9: Polls vs prediction markets (captures the comparison directly)
COMPARISON = [
    "polls vs prediction markets",
    "polls vs betting odds",
    "polls vs polymarket",
    "polls wrong prediction markets right",
    "prediction markets more accurate than polls",
]

ALL_KEYWORDS = (
    SOURCE_POLITICAL + GENERIC_POLITICAL + CANDIDATE_POLLS
    + EVENT_SPECIFIC + TIME_SPECIFIC + RACE_POLLS
    + HASHTAG_SEARCHES + METHODOLOGY + COMPARISON
)


# ---------------------------------------------------------------------------
# Collection logic (mirrors tiktok/ and youtube/ pipelines)
# ---------------------------------------------------------------------------

def collect(platform: str):
    """Run Bright Data collection for the given platform."""
    if not BRIGHTDATA_API_TOKEN:
        print("ERROR: Set BRIGHTDATA_API_TOKEN environment variable")
        sys.exit(1)

    if platform == "tiktok":
        dataset_id = TIKTOK_DATASET_ID
        output_path = TIKTOK_RAW
        keyword_field = "search_keyword"
        limit_per_input = 500
        max_wait = 30 * 60
    elif platform == "youtube":
        dataset_id = YOUTUBE_DATASET_ID
        output_path = YOUTUBE_RAW
        keyword_field = "keyword"
        limit_per_input = 200
        max_wait = 60 * 60
    else:
        print(f"ERROR: Unknown platform '{platform}'")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {BRIGHTDATA_API_TOKEN}",
        "Content-Type": "application/json",
    }
    base_url = "https://api.brightdata.com/datasets/v3"

    print(f"\n{'='*60}")
    print(f"  Collecting {platform.upper()} poll-referencing videos")
    print(f"  {len(ALL_KEYWORDS)} keywords x {limit_per_input} results each")
    print(f"  Dataset: {dataset_id}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    # --- Step 1: Fire all triggers ---
    print(f"\n  Firing {len(ALL_KEYWORDS)} triggers...")
    jobs = {}  # snapshot_id -> keyword

    for i, kw in enumerate(ALL_KEYWORDS):
        resp = requests.post(
            f"{base_url}/trigger",
            headers=headers,
            json=[{keyword_field: kw}],
            params={
                "dataset_id": dataset_id,
                "format": "json",
                "type": "discover_new",
                "discover_by": "keyword",
                "limit_per_input": limit_per_input,
            },
        )
        if resp.status_code != 200:
            print(f"    [{i+1}] ERROR '{kw}': {resp.status_code} {resp.text[:200]}")
            continue

        snapshot_id = resp.json().get("snapshot_id")
        if snapshot_id:
            jobs[snapshot_id] = kw
            print(f"    [{i+1}/{len(ALL_KEYWORDS)}] '{kw}' -> {snapshot_id}")
        else:
            print(f"    [{i+1}] ERROR '{kw}': no snapshot_id")

        time.sleep(0.5)

    print(f"\n  Triggered {len(jobs)}/{len(ALL_KEYWORDS)} jobs. Polling for results...")

    # Save job mapping for resume
    jobs_file = output_path.with_suffix(".jobs.json")
    with open(jobs_file, "w") as f:
        json.dump(jobs, f, indent=2)

    # --- Step 2: Poll all jobs until done ---
    pending = dict(jobs)
    all_results = []
    start_time = time.time()

    while pending and (time.time() - start_time) < max_wait:
        time.sleep(15)
        elapsed = int(time.time() - start_time)

        newly_done = []
        for snapshot_id, kw in list(pending.items()):
            try:
                status_resp = requests.get(
                    f"{base_url}/progress/{snapshot_id}",
                    headers=headers,
                )
                progress = status_resp.json()
                status = progress.get("status")
            except Exception as e:
                print(f"    Poll error for {snapshot_id}: {e}")
                continue

            if status == "ready":
                records = progress.get("records", 0)
                newly_done.append(snapshot_id)

                if records > 0:
                    data_resp = requests.get(
                        f"{base_url}/snapshot/{snapshot_id}",
                        headers=headers,
                        params={"format": "json"},
                    )
                    if data_resp.status_code == 200:
                        results = data_resp.json()
                        for r in results:
                            r["_search_keyword"] = kw
                        all_results.extend(results)
                        print(f"    [{elapsed}s] '{kw}': {records} records downloaded")
                    else:
                        print(f"    [{elapsed}s] '{kw}': download error {data_resp.status_code}")
                else:
                    print(f"    [{elapsed}s] '{kw}': 0 records")

            elif status == "failed":
                newly_done.append(snapshot_id)
                print(f"    [{elapsed}s] '{kw}': FAILED")

        for sid in newly_done:
            del pending[sid]

        if newly_done:
            _save_deduped(all_results, output_path, platform)
            done = len(jobs) - len(pending)
            print(f"    --- {done}/{len(jobs)} complete, {len(pending)} pending ---")

    if pending:
        print(f"\n  WARNING: {len(pending)} jobs timed out after {max_wait//60} min:")
        for sid, kw in pending.items():
            print(f"    '{kw}' ({sid})")

    deduped = _save_deduped(all_results, output_path, platform)
    elapsed = int(time.time() - start_time)
    print(f"\n  Collection complete in {elapsed//60}m {elapsed%60}s")
    print(f"  {len(deduped)} unique videos (from {len(all_results)} total across {len(jobs)} queries)")
    return deduped


def _save_deduped(all_results: list, output_path: Path, platform: str) -> list:
    """Deduplicate by video ID and merge with existing data."""
    seen = set()
    deduped = []

    if output_path.exists():
        try:
            existing = json.load(open(output_path, encoding="utf-8"))
            for r in existing:
                vid = _get_id(r, platform)
                if vid and vid not in seen:
                    seen.add(vid)
                    deduped.append(r)
        except (json.JSONDecodeError, IOError):
            pass

    for r in all_results:
        vid = _get_id(r, platform)
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(r)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    return deduped


def _get_id(record: dict, platform: str) -> str:
    if platform == "tiktok":
        return record.get("post_id") or record.get("video_id") or record.get("id") or record.get("url", "")
    else:
        return record.get("video_id") or record.get("url") or record.get("id", "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect poll-referencing videos from TikTok and YouTube"
    )
    parser.add_argument(
        "platform",
        choices=["tiktok", "youtube", "both"],
        help="Which platform to collect from",
    )
    args = parser.parse_args()

    print(f"\nPoll Video Collection")
    print(f"Keywords: {len(ALL_KEYWORDS)} total")
    print(f"  Source-specific: {len(SOURCE_POLITICAL)}")
    print(f"  Generic political: {len(GENERIC_POLITICAL)}")
    print(f"  Candidate + polls: {len(CANDIDATE_POLLS)}")
    print(f"  Event-specific: {len(EVENT_SPECIFIC)}")
    print(f"  Time-specific: {len(TIME_SPECIFIC)}")
    print(f"  Race-specific: {len(RACE_POLLS)}")
    print(f"  Hashtags: {len(HASHTAG_SEARCHES)}")
    print(f"  Methodology: {len(METHODOLOGY)}")
    print(f"  PM comparison: {len(COMPARISON)}")

    if args.platform in ("tiktok", "both"):
        collect("tiktok")
    if args.platform in ("youtube", "both"):
        collect("youtube")


if __name__ == "__main__":
    main()
