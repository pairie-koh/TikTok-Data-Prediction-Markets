"""
Two-pass filter for YouTube poll-referencing videos.

Pass 1 (keyword, free): Keep only videos that explicitly reference polling
        in title, description, hashtags, or transcript.

Pass 2 (LLM, cheap): For each survivor, ask whether the video substantively
        discusses US political polls or is noise/irrelevant.

Usage:
    python polls/scripts/filter_youtube_polls.py --keyword-only
    python polls/scripts/filter_youtube_polls.py
    python polls/scripts/filter_youtube_polls.py --stats
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
POLLS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = POLLS_DIR / "data"
RAW_JSON = DATA_DIR / "raw" / "youtube_polls_raw.json"
OUTPUT_CSV = DATA_DIR / "youtube_polls_filtered.csv"
PROGRESS_FILE = DATA_DIR / "youtube_filter_llm_progress.json"

# ---------------------------------------------------------------------------
# API setup
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

# ---------------------------------------------------------------------------
# Pass 1: Polling keyword patterns (same as TikTok filter)
# ---------------------------------------------------------------------------

SOURCE_PATTERNS = [
    r"five\s*thirty\s*eight",
    r"538",
    r"real\s*clear\s*politics",
    r"rcp\b",
    r"nate\s+silver",
    r"quinnipiac",
    r"rasmussen",
    r"trafalgar",
    r"monmouth",
    r"marist",
    r"emerson\s+poll",
    r"morning\s+consult",
    r"gallup",
    r"pew\s+research",
    r"ipsos",
    r"yougov",
    r"siena\s+poll",
    r"fox\s+news\s+poll",
    r"cnn\s+poll",
    r"nbc\s+poll",
    r"abc\s+poll",
    r"cbs\s+poll",
    r"nyt\s+poll",
    r"new\s+york\s+times\s+poll",
    r"reuters\s+poll",
]

GENERIC_PATTERNS = [
    r"\bpolls?\b",
    r"\bpolling\b",
    r"poll\s+numbers?",
    r"poll\s+results?",
    r"poll\s+average",
    r"polling\s+data",
    r"polling\s+average",
    r"approval\s+rat(?:ing|e)",
    r"favorab(?:ility|le)\s+rat(?:ing|e)",
    r"exit\s+polls?",
    r"tracking\s+poll",
    r"likely\s+voters?",
    r"registered\s+voters?",
    r"margin\s+of\s+error",
    r"sample\s+size",
    r"poll\s+tracker",
    r"swing\s+state\s+polls?",
    r"battleground\s+(?:state\s+)?polls?",
]

SOURCE_RE = re.compile("|".join(SOURCE_PATTERNS), re.IGNORECASE)
GENERIC_RE = re.compile("|".join(GENERIC_PATTERNS), re.IGNORECASE)

# ---------------------------------------------------------------------------
# Pass 2: LLM prompt
# ---------------------------------------------------------------------------

POLLS_PROMPT = """A YouTube video was found by searching for poll-related keywords. Does this video substantively discuss US political polls or polling data?

YES — The video substantively discusses US political polls. This includes:
- Citing specific poll numbers, poll averages, or approval ratings for US candidates
- Discussing who is ahead/behind in polls for US elections (President, Senate, House, Governor)
- Analyzing polling methodology, accuracy, or trends in US elections
- Comparing polls to prediction markets or betting odds for US races
- Commentary on swing state polls, battleground state polls, or national polls
- Discussing pollsters, polling aggregators (538, RCP), or poll trackers
- Reacting to new poll releases about US political races

NO — The video does NOT substantively discuss US political polls. This includes:
- Video mentions "polls" only in passing with no real discussion
- Sports content, entertainment, or non-political topics
- Foreign elections or non-US political polls
- General political commentary that doesn't reference actual poll data
- "Who do you think will win?" without citing real polls
- Platform tutorials, trading advice, or crypto content
- Videos where the search keyword match is incidental / irrelevant to actual content

Respond with exactly one word: YES or NO

---
Title: {title}
Description: {description}
Transcript: {transcript}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_llm_client():
    from openai import OpenAI

    if OPENROUTER_API_KEY:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY), LLM_MODEL
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), "gpt-4o-mini"
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY), "claude-sonnet-4-20250514"
        except ImportError:
            pass
    print("ERROR: Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
    sys.exit(1)


def llm_call(client, model: str, prompt: str, max_tokens: int = 10) -> str:
    try:
        import anthropic
        is_anthropic = isinstance(client, anthropic.Anthropic)
    except ImportError:
        is_anthropic = False

    if is_anthropic:
        resp = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    else:
        resp = client.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()


def get_transcript(video: dict) -> str:
    """Extract transcript from YouTube video record."""
    t = video.get("transcript", "")
    if t and isinstance(t, str):
        return t.strip()
    return ""


def find_poll_mentions(text: str) -> dict:
    sources = list(set(SOURCE_RE.findall(text)))
    generic = list(set(GENERIC_RE.findall(text)))
    return {"sources": sources, "generic": generic}


# ---------------------------------------------------------------------------
# Pass 1
# ---------------------------------------------------------------------------

def pass1_keyword_filter(raw_videos: list[dict]) -> list[dict]:
    seen = set()
    deduped = []
    for v in raw_videos:
        vid = v.get("video_id") or v.get("url", "")
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(v)
    print(f"  After dedup: {len(deduped)} unique videos")

    matched = []
    no_transcript = 0
    source_match = 0
    generic_only = 0

    for v in deduped:
        title = v.get("title", "") or ""
        description = v.get("description", "") or ""
        hashtags_raw = v.get("hashtags", [])
        if isinstance(hashtags_raw, list):
            hashtags = " ".join(str(h.get("name", "")) if isinstance(h, dict) else str(h) for h in hashtags_raw)
        else:
            hashtags = str(hashtags_raw or "")
        tags_raw = v.get("tags", [])
        if isinstance(tags_raw, list):
            tags = " ".join(str(t.get("name", "")) if isinstance(t, dict) else str(t) for t in tags_raw)
        else:
            tags = str(tags_raw or "")
        transcript = get_transcript(v)

        if not transcript:
            no_transcript += 1

        all_text = f"{title} {description} {hashtags} {tags} {transcript}"
        mentions = find_poll_mentions(all_text)

        has_source = len(mentions["sources"]) > 0
        has_generic = len(mentions["generic"]) > 0

        if has_source or has_generic:
            found_in = []
            if find_poll_mentions(f"{title} {description}")["sources"] or find_poll_mentions(f"{title} {description}")["generic"]:
                found_in.append("title_description")
            if find_poll_mentions(f"{hashtags} {tags}")["sources"] or find_poll_mentions(f"{hashtags} {tags}")["generic"]:
                found_in.append("hashtags_tags")
            if transcript and (find_poll_mentions(transcript)["sources"] or find_poll_mentions(transcript)["generic"]):
                found_in.append("transcript")

            v["_match_type"] = "source" if has_source else "generic_only"
            v["_match_sources"] = ", ".join(mentions["sources"]) if mentions["sources"] else ""
            v["_match_source"] = ", ".join(found_in)
            v["_transcript_text"] = transcript
            matched.append(v)

            if has_source:
                source_match += 1
            else:
                generic_only += 1

    print(f"  No transcript available: {no_transcript}")
    print(f"  Pass 1 result: {len(matched)} mention polling")
    print(f"    Named source match: {source_match}")
    print(f"    Generic poll term only: {generic_only}")

    return matched


# ---------------------------------------------------------------------------
# Pass 2
# ---------------------------------------------------------------------------

def pass2_llm_classify(videos: list[dict], workers: int = 10) -> list[dict]:
    client, model = get_llm_client()
    print(f"  Using model: {model}")
    print(f"  Videos to classify: {len(videos)}")
    print(f"  Workers: {workers}")

    already_done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already classified")

    # Assign cached results and collect work items
    to_classify = []
    for v in videos:
        vid = v.get("video_id") or v.get("url", "")
        if vid in already_done:
            v["_topic"] = already_done[vid]
        else:
            to_classify.append(v)

    print(f"  Need to classify: {len(to_classify)}")
    if not to_classify:
        topic_counts = Counter(v.get("_topic", "UNKNOWN") for v in videos)
        for t, c in topic_counts.most_common():
            print(f"    {t}: {c} ({c/len(videos)*100:.1f}%)")
        return videos

    lock = threading.Lock()
    api_calls = 0
    errors = 0
    rate_semaphore = threading.Semaphore(workers)

    def classify_one(v):
        nonlocal api_calls, errors
        vid = v.get("video_id") or v.get("url", "")
        title = (v.get("title", "") or "")[:200]
        description = (v.get("description", "") or "")[:500]
        transcript = (v.get("_transcript_text", "") or "")[:1500]
        prompt = POLLS_PROMPT.format(title=title, description=description, transcript=transcript)

        result = "UNKNOWN"
        for attempt in range(5):
            try:
                with rate_semaphore:
                    resp = llm_call(client, model, prompt, max_tokens=10).upper().strip()
                if resp not in ("YES", "NO"):
                    if "YES" in resp:
                        resp = "YES"
                    elif "NO" in resp:
                        resp = "NO"
                    else:
                        resp = "UNKNOWN"
                result = resp
                with lock:
                    api_calls += 1
                break
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e):
                    wait = 2 ** attempt + 1
                    time.sleep(wait)
                    continue
                print(f"    Error on {vid}: {e}")
                with lock:
                    errors += 1
                break

        v["_topic"] = result
        with lock:
            already_done[vid] = result
        return vid, result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(classify_one, v): v for v in to_classify}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            future.result()
            if done_count % 50 == 0:
                with lock:
                    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                        json.dump(already_done, f)
                    yes_count = sum(1 for t in already_done.values() if t == "YES")
                    no_count = sum(1 for t in already_done.values() if t == "NO")
                    print(f"    [{done_count}/{len(to_classify)}] API calls: {api_calls} | YES: {yes_count}, NO: {no_count}")

    # Final save
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    print(f"\n  Pass 2 complete. API calls: {api_calls}, errors: {errors}")
    topic_counts = Counter(v.get("_topic", "UNKNOWN") for v in videos)
    for t, c in topic_counts.most_common():
        print(f"    {t}: {c} ({c/len(videos)*100:.1f}%)")

    return videos


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_output(videos: list[dict]):
    fieldnames = [
        "video_id", "url", "title", "description", "date_posted",
        "likes", "views", "num_comments", "video_length",
        "youtuber", "channel_url", "subscribers", "verified",
        "_search_keyword", "_match_type", "_match_sources",
        "_match_source", "_topic", "_transcript_text",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in videos:
            row = {**v}
            if isinstance(row.get("hashtags"), list):
                row["hashtags"] = ", ".join(
                    str(h.get("name", "")) if isinstance(h, dict) else str(h)
                    for h in row["hashtags"]
                )
            writer.writerow(row)

    print(f"\nSaved {len(videos)} videos to {OUTPUT_CSV}")


def print_stats():
    if not RAW_JSON.exists():
        print("No raw data found.")
        return

    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    has_transcript = sum(1 for v in raw if get_transcript(v))
    print(f"Raw YouTube polls videos: {len(raw)}")
    print(f"Have transcript:          {has_transcript}")
    print(f"No transcript:            {len(raw) - has_transcript}")

    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        yes_count = sum(1 for t in progress.values() if t == "YES")
        no_count = sum(1 for t in progress.values() if t == "NO")
        print(f"\nLLM classifications:  {len(progress)}")
        print(f"  YES (polls):        {yes_count}")
        print(f"  NO (not polls):     {no_count}")

    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"\nFiltered CSV:         {len(rows)} rows")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter YouTube polls videos")
    parser.add_argument("--keyword-only", action="store_true",
                        help="Run only the keyword pass (no LLM)")
    parser.add_argument("--stats", action="store_true",
                        help="Print stats and exit")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    print("Loading raw YouTube polls data...")
    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Raw videos: {len(raw)}")

    print("\n=== PASS 1: Keyword filter (polling terms) ===")
    survivors = pass1_keyword_filter(raw)

    if args.keyword_only:
        for v in survivors:
            v["_topic"] = ""
        save_output(survivors)
        return

    print("\n=== PASS 2: LLM classification (substantive polls discussion) ===")
    classified = pass2_llm_classify(survivors)

    save_output(classified)

    yes_vids = [v for v in classified if v.get("_topic") == "YES"]
    no_vids = [v for v in classified if v.get("_topic") == "NO"]
    total_views = sum(int(v.get("views", 0) or 0) for v in classified)
    yes_views = sum(int(v.get("views", 0) or 0) for v in yes_vids)
    no_views = sum(int(v.get("views", 0) or 0) for v in no_vids)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  SUBSTANTIVE POLLS (YES): {len(yes_vids)} videos, {yes_views:,} views")
    print(f"  NOT POLLS (NO):          {len(no_vids)} videos, {no_views:,} views")
    print(f"  Total views:             {total_views:,}")


if __name__ == "__main__":
    main()
