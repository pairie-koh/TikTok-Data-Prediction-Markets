"""
Two-pass filter for YouTube prediction market videos.

Pass 1 (keyword, free): Keep only videos that explicitly name a prediction market
        platform (Polymarket, Kalshi, PredictIt) in title, description, hashtags,
        or transcript.

Pass 2 (LLM, cheap): For each survivor, ask whether the video is about US politics
        or purely about trading/crypto/sports/the platform as a business.

Usage:
    # Set one of these:
    export OPENROUTER_API_KEY=...
    export ANTHROPIC_API_KEY=...

    # Run both passes:
    python youtube/scripts/filter_by_platform.py

    # Run only keyword pass (no LLM needed):
    python youtube/scripts/filter_by_platform.py --keyword-only
"""

import argparse
import csv
import json
import os
import re
import sys
import io
import time
from collections import Counter
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_JSON = DATA_DIR / "raw" / "youtube_raw.json"
OUTPUT_CSV = DATA_DIR / "youtube_platform_filtered.csv"
PROGRESS_FILE = DATA_DIR / "youtube_filter_llm_progress.json"

# --- API setup ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

# --- Platform keyword patterns (case-insensitive) ---
PLATFORM_PATTERNS = [
    r"polymarket",
    r"poly[\s\-]?market",
    r"kalshi",
    r"predictit",
    r"predict[\s\-]?it",
    r"prediction\s+markets?",
    r"betting\s+markets?",
]
PLATFORM_RE = re.compile("|".join(PLATFORM_PATTERNS), re.IGNORECASE)

# --- LLM prompt for pass 2 ---
POLITICS_PROMPT = """A YouTube video mentions a prediction market platform ({platforms}). Does this video discuss US politics specifically?

YES — The video is about US politics. This includes:
- US election odds, US candidate chances (Trump, Harris, Biden, Vance, DeSantis, etc.)
- US Senate, House, Governor, or Presidential race predictions
- Betting on US political outcomes (even if framed as trading advice)
- US government policy predictions (tariffs, Fed rate cuts, US regulation)
- Commentary on US political events using prediction market odds
- Discussion of prediction market regulation by US government (CFTC, Congress)

NO — The video is NOT about US politics. This includes:
- Sports betting (football, soccer, boxing, basketball, UFC, etc.)
- Foreign wars, geopolitical conflicts, or non-US events (even if political)
- Non-US elections or foreign government predictions
- Weather, entertainment, celebrity, or non-political predictions
- General platform tutorials, trading advice, or "how to make money" content
- Crypto/finance content unrelated to US political outcomes
- Sponsored content, ads, or promo codes with no US political substance
- Prediction markets as a business/phenomenon without US political context

Respond with exactly one word: YES or NO

---
Title: {title}
Description: {description}
Transcript: {transcript}"""


def get_llm_client():
    from openai import OpenAI

    if OPENROUTER_API_KEY:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY), LLM_MODEL
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), "gpt-4o-mini"
    if ANTHROPIC_API_KEY:
        try:
            import anthropic
            return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY), "claude-haiku-4-20250414"
        except ImportError:
            pass
    print("ERROR: Set OPENROUTER_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY")
    sys.exit(1)


def llm_call(client, model, prompt, max_tokens=10):
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


def get_transcript(video):
    """Extract transcript text from a YouTube video record."""
    # Try formatted_transcript first (list of dicts with 'text')
    ft = video.get("formatted_transcript")
    if ft and isinstance(ft, list) and len(ft) > 0:
        parts = []
        for item in ft:
            if isinstance(item, dict):
                parts.append(item.get("text", "") or "")
            elif isinstance(item, str):
                parts.append(item)
        text = " ".join(parts).strip()
        if text:
            return text

    # Fall back to transcript field (raw string)
    t = video.get("transcript", "")
    if t and isinstance(t, str) and len(t) > 10:
        return t.strip()

    return ""


def find_platforms(text):
    matches = PLATFORM_RE.findall(text)
    platforms = set()
    for m in matches:
        m_lower = m.lower().replace(" ", "").replace("-", "")
        if "polymarket" in m_lower:
            platforms.add("polymarket")
        elif "kalshi" in m_lower:
            platforms.add("kalshi")
        elif "predictit" in m_lower:
            platforms.add("predictit")
        elif "predictionmarket" in m_lower:
            platforms.add("prediction_market")
        elif "bettingmarket" in m_lower:
            platforms.add("betting_market")
    return sorted(platforms)


def pass1_keyword_filter(raw_videos):
    """Pass 1: Keep only videos that mention a platform by name."""
    # Deduplicate by video_id
    seen = set()
    deduped = []
    for v in raw_videos:
        vid = v.get("video_id", "")
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(v)
    print(f"  After dedup: {len(deduped)} unique videos")

    matched = []
    no_transcript = 0

    for v in deduped:
        title = v.get("title", "") or ""
        description = v.get("description", "") or ""
        hashtags_raw = v.get("hashtags", [])
        if isinstance(hashtags_raw, list):
            parts = []
            for h in hashtags_raw:
                if isinstance(h, dict):
                    parts.append(h.get("hashtag", "") or h.get("name", ""))
                else:
                    parts.append(str(h))
            hashtags = " ".join(parts)
        else:
            hashtags = str(hashtags_raw or "")
        transcript = get_transcript(v)

        if not transcript:
            no_transcript += 1

        # Check each source
        found_in = []
        if find_platforms(title):
            found_in.append("title")
        if find_platforms(description):
            found_in.append("description")
        if find_platforms(hashtags):
            found_in.append("hashtags")
        if find_platforms(transcript):
            found_in.append("transcript")

        all_text = f"{title} {description} {hashtags} {transcript}"
        platforms = find_platforms(all_text)

        if platforms:
            v["_platforms"] = ", ".join(platforms)
            v["_match_source"] = ", ".join(found_in)
            v["_transcript_text"] = transcript
            matched.append(v)

    print(f"  No transcript available: {no_transcript}")
    print(f"  Pass 1 result: {len(matched)} mention a platform by name")

    # Breakdown
    plat_counts = Counter()
    for v in matched:
        for p in v["_platforms"].split(", "):
            plat_counts[p] += 1
    for p, c in plat_counts.most_common():
        print(f"    {p}: {c}")

    source_counts = Counter()
    for v in matched:
        for s in v["_match_source"].split(", "):
            source_counts[s] += 1
    print(f"  Match sources:")
    for s, c in source_counts.most_common():
        print(f"    {s}: {c}")

    transcript_only = sum(1 for v in matched if v["_match_source"] == "transcript")
    print(f"  Matched ONLY via transcript: {transcript_only}")

    return matched


def pass2_llm_classify(videos):
    """Pass 2: Use LLM to classify each video as political or not."""
    client, model = get_llm_client()
    print(f"  Using model: {model}")
    print(f"  Videos to classify: {len(videos)}")

    already_done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already classified")

    api_calls = 0
    errors = 0

    for i, v in enumerate(videos):
        vid = v.get("video_id", "")

        if vid in already_done:
            v["_topic"] = already_done[vid]
            continue

        title = (v.get("title", "") or "")[:300]
        description = (v.get("description", "") or "")[:500]
        transcript = (v.get("_transcript_text", "") or "")[:1500]
        platforms = v.get("_platforms", "")

        prompt = POLITICS_PROMPT.format(
            platforms=platforms, title=title,
            description=description, transcript=transcript,
        )

        try:
            result = llm_call(client, model, prompt, max_tokens=10).upper().strip()
            if result not in ("YES", "NO"):
                if "YES" in result:
                    result = "YES"
                elif "NO" in result:
                    result = "NO"
                else:
                    result = "UNKNOWN"
            api_calls += 1
        except Exception as e:
            print(f"    Error on {vid}: {e}")
            result = "UNKNOWN"
            errors += 1

        v["_topic"] = result
        already_done[vid] = result

        if (api_calls + errors) % 25 == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(already_done, f)
            yes_count = sum(1 for t in already_done.values() if t == "YES")
            no_count = sum(1 for t in already_done.values() if t == "NO")
            print(f"    [{i+1}/{len(videos)}] API calls: {api_calls} | YES: {yes_count}, NO: {no_count}")

        time.sleep(0.15)

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    print(f"\n  Pass 2 complete. API calls: {api_calls}, errors: {errors}")
    topic_counts = Counter(v.get("_topic", "UNKNOWN") for v in videos)
    for t, c in topic_counts.most_common():
        print(f"    {t}: {c} ({c/len(videos)*100:.1f}%)")

    return videos


def save_output(videos):
    """Save filtered CSV."""
    fieldnames = [
        "video_id", "url", "title", "description", "date_posted",
        "likes", "views", "num_comments", "video_length",
        "youtuber", "channel_url", "subscribers", "verified",
        "_search_keyword", "_platforms", "_match_source", "_topic",
        "_transcript_text",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in videos:
            row = {**v}
            if isinstance(row.get("hashtags"), list):
                parts = []
                for h in row["hashtags"]:
                    if isinstance(h, dict):
                        parts.append(h.get("hashtag", "") or h.get("name", ""))
                    else:
                        parts.append(str(h))
                row["hashtags"] = ", ".join(parts)
            writer.writerow(row)

    print(f"\nSaved {len(videos)} videos to {OUTPUT_CSV}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-only", action="store_true",
                        help="Run only the keyword pass (no LLM)")
    args = parser.parse_args()

    print("Loading raw data...")
    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Raw videos: {len(raw)}")

    # Pass 1: keyword filter
    print("\n=== PASS 1: Keyword filter (platform names) ===")
    survivors = pass1_keyword_filter(raw)

    if args.keyword_only:
        for v in survivors:
            v["_topic"] = ""
        save_output(survivors)
        return

    # Pass 2: LLM topic classification
    print("\n=== PASS 2: LLM classification (political vs not) ===")
    classified = pass2_llm_classify(survivors)

    save_output(classified)

    # Summary
    yes_vids = [v for v in classified if v.get("_topic") == "YES"]
    no_vids = [v for v in classified if v.get("_topic") == "NO"]
    total_views = sum(int(v.get("views", 0) or 0) for v in classified)
    yes_views = sum(int(v.get("views", 0) or 0) for v in yes_vids)
    no_views = sum(int(v.get("views", 0) or 0) for v in no_vids)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  POLITICAL (YES): {len(yes_vids)} videos, {yes_views:,} views")
    print(f"  NOT POLITICAL (NO): {len(no_vids)} videos, {no_views:,} views")


if __name__ == "__main__":
    main()
