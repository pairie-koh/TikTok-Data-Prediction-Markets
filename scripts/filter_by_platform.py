"""
Two-pass filter for TikTok prediction market videos.

Pass 1 (keyword, free): Keep only videos that explicitly name a prediction market
        platform (Polymarket, Kalshi, PredictIt) in description, hashtags, or transcript.

Pass 2 (LLM, cheap): For each survivor, ask whether the video is about politics/elections
        or purely about trading/crypto/the platform as a business.

Usage:
    # Set one of these:
    export ANTHROPIC_API_KEY=...
    export OPENROUTER_API_KEY=...

    # Run both passes:
    python scripts/filter_by_platform.py

    # Run only keyword pass (no LLM needed):
    python scripts/filter_by_platform.py --keyword-only
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_JSON = DATA_DIR / "raw" / "tiktok_raw.json"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
TIKTOK_DIR = DATA_DIR / "tiktok_transcripts"
CODED_CSV = DATA_DIR / "tiktok_coded.csv"
OUTPUT_CSV = DATA_DIR / "tiktok_platform_filtered.csv"
PROGRESS_FILE = DATA_DIR / "filter_llm_progress.json"

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
]
PLATFORM_RE = re.compile("|".join(PLATFORM_PATTERNS), re.IGNORECASE)

# --- LLM prompt for pass 2 ---
POLITICS_PROMPT = """A TikTok video mentions a prediction market platform ({platforms}). Does this video discuss US politics specifically?

YES — The video is about US politics. This includes:
- US election odds, US candidate chances (Trump, Harris, Biden, Vance, DeSantis, etc.)
- US Senate, House, Governor, or Presidential race predictions
- Betting on US political outcomes (even if framed as trading advice)
- US government policy predictions (tariffs, Fed rate cuts, US regulation)
- Commentary on US political events using prediction market odds
- Discussion of prediction market regulation by US government (CFTC, Congress)

NO — The video is NOT about US politics. This includes:
- Sports betting (football, soccer, boxing, basketball, UFC, La Liga, UCL, etc.)
- Foreign wars, geopolitical conflicts, or non-US events (even if political)
- Non-US elections or foreign government predictions
- Weather, entertainment, celebrity, or non-political predictions
- General platform tutorials, trading advice, or "how to make money" content
- Crypto/finance content unrelated to US political outcomes
- Sponsored content, ads, or promo codes with no US political substance
- Prediction markets as a business/phenomenon without US political context

Respond with exactly one word: YES or NO

---
Description: {description}
Transcript: {transcript}"""


def get_llm_client():
    """Get an OpenAI-compatible client."""
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


def llm_call(client, model: str, prompt: str, max_tokens: int = 10) -> str:
    """Unified LLM call for OpenAI-compatible and Anthropic clients."""
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


def get_transcript(post_id: str) -> str:
    """Load transcript: whisper > tiktok captions > coded CSV."""
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

    return ""


def get_csv_transcripts() -> dict[str, str]:
    """Load transcripts embedded in the coded CSV as last-resort fallback."""
    transcripts = {}
    if not CODED_CSV.exists():
        return transcripts
    with open(CODED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("transcript", "").strip()
            if t:
                if t.startswith("{") or t.startswith("["):
                    try:
                        obj = json.loads(t)
                        if isinstance(obj, dict) and "utterances" in obj:
                            t = " ".join(u["text"] for u in obj["utterances"] if u.get("text"))
                        elif isinstance(obj, list):
                            t = " ".join(item.get("text", "") for item in obj if isinstance(item, dict))
                    except (json.JSONDecodeError, KeyError):
                        pass
                transcripts[row["post_id"]] = t
    return transcripts


def find_platforms(text: str) -> list[str]:
    """Return canonical platform names found in text."""
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
    return sorted(platforms)


def pass1_keyword_filter(raw_videos: list[dict], csv_transcripts: dict) -> list[dict]:
    """Pass 1: Keep only videos that mention a platform by name."""
    # Deduplicate
    seen = set()
    deduped = []
    for v in raw_videos:
        pid = str(v.get("post_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            deduped.append(v)
    print(f"  After dedup: {len(deduped)} unique videos")

    matched = []
    no_transcript = 0

    for v in deduped:
        pid = str(v.get("post_id", ""))
        description = v.get("description", "") or ""
        hashtags_raw = v.get("hashtags", [])
        hashtags = " ".join(hashtags_raw) if isinstance(hashtags_raw, list) else str(hashtags_raw)

        transcript = get_transcript(pid)
        if not transcript:
            transcript = csv_transcripts.get(pid, "")
        if not transcript:
            no_transcript += 1

        # Check each source
        found_in = []
        if find_platforms(description):
            found_in.append("description")
        if find_platforms(hashtags):
            found_in.append("hashtags")
        if find_platforms(transcript):
            found_in.append("transcript")

        all_text = f"{description} {hashtags} {transcript}"
        platforms = find_platforms(all_text)

        if platforms:
            v["_platforms"] = ", ".join(platforms)
            v["_match_source"] = ", ".join(found_in)
            v["_transcript"] = transcript
            matched.append(v)

    print(f"  No transcript available: {no_transcript}")
    print(f"  Pass 1 result: {len(matched)} mention a platform by name")

    # Breakdown
    from collections import Counter
    plat_counts = Counter()
    for v in matched:
        for p in v["_platforms"].split(", "):
            plat_counts[p] += 1
    for p, c in plat_counts.most_common():
        print(f"    {p}: {c}")

    transcript_only = sum(1 for v in matched if v["_match_source"] == "transcript")
    print(f"  Matched ONLY via transcript: {transcript_only}")

    return matched


def pass2_llm_classify(videos: list[dict]) -> list[dict]:
    """Pass 2: Use LLM to classify each video as POLITICAL, TRADING, or META."""
    client, model = get_llm_client()
    print(f"  Using model: {model}")
    print(f"  Videos to classify: {len(videos)}")

    # Resume from progress
    already_done = {}
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            already_done = json.load(f)
        print(f"  Resuming: {len(already_done)} already classified")

    api_calls = 0
    errors = 0

    for i, v in enumerate(videos):
        pid = str(v.get("post_id", ""))

        if pid in already_done:
            v["_topic"] = already_done[pid]
            continue

        description = (v.get("description", "") or "")[:500]
        transcript = (v.get("_transcript", "") or "")[:1500]
        platforms = v.get("_platforms", "")

        prompt = POLITICS_PROMPT.format(
            platforms=platforms,
            description=description,
            transcript=transcript,
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
            print(f"    Error on {pid}: {e}")
            result = "UNKNOWN"
            errors += 1

        v["_topic"] = result
        already_done[pid] = result

        # Save progress every 25
        if (api_calls + errors) % 25 == 0:
            with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(already_done, f)
            yes_count = sum(1 for t in already_done.values() if t == "YES")
            no_count = sum(1 for t in already_done.values() if t == "NO")
            print(f"    [{i+1}/{len(videos)}] API calls: {api_calls} | YES: {yes_count}, NO: {no_count}")

        time.sleep(0.15)  # rate limit

    # Final save
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(already_done, f)

    print(f"\n  Pass 2 complete. API calls: {api_calls}, errors: {errors}")
    from collections import Counter
    topic_counts = Counter(v.get("_topic", "UNKNOWN") for v in videos)
    for t, c in topic_counts.most_common():
        print(f"    {t}: {c} ({c/len(videos)*100:.1f}%)")

    return videos


def save_output(videos: list[dict]):
    """Save final filtered CSV."""
    fieldnames = [
        "post_id", "url", "description", "create_time",
        "digg_count", "share_count", "collect_count", "comment_count",
        "play_count", "video_duration", "hashtags",
        "profile_username", "profile_followers", "is_verified", "region",
        "_search_keyword", "_platforms", "_match_source", "_topic", "_transcript",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in videos:
            row = {**v}
            if isinstance(row.get("hashtags"), list):
                row["hashtags"] = ", ".join(row["hashtags"])
            writer.writerow(row)

    print(f"\nSaved {len(videos)} videos to {OUTPUT_CSV}")


def print_comparison(videos: list[dict]):
    """Compare with old coded dataset."""
    if not CODED_CSV.exists():
        return
    coded_ids = set()
    with open(CODED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            coded_ids.add(row["post_id"])

    new_ids = set(str(v["post_id"]) for v in videos)
    overlap = coded_ids & new_ids
    old_only = coded_ids - new_ids
    new_only = new_ids - coded_ids

    print(f"\n=== COMPARISON WITH OLD CODED DATASET ({len(coded_ids)} videos) ===")
    print(f"  Overlap:               {len(overlap)}")
    print(f"  Old only (would drop): {len(old_only)}")
    print(f"  New finds:             {len(new_only)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword-only", action="store_true",
                        help="Run only the keyword pass (no LLM)")
    args = parser.parse_args()

    # Load raw data
    print("Loading raw data...")
    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  Raw videos: {len(raw)}")

    csv_transcripts = get_csv_transcripts()
    print(f"  CSV transcript fallbacks: {len(csv_transcripts)}")

    # Pass 1: keyword filter
    print("\n=== PASS 1: Keyword filter (platform names) ===")
    survivors = pass1_keyword_filter(raw, csv_transcripts)

    if args.keyword_only:
        # Save without LLM topic column
        for v in survivors:
            v["_topic"] = ""
        save_output(survivors)
        print_comparison(survivors)
        return

    # Pass 2: LLM topic classification
    print("\n=== PASS 2: LLM classification (political vs trading vs meta) ===")
    classified = pass2_llm_classify(survivors)

    # Save all (user can filter by _topic later)
    save_output(classified)
    print_comparison(classified)

    # Summary
    yes_vids = [v for v in classified if v.get("_topic") == "YES"]
    no_vids = [v for v in classified if v.get("_topic") == "NO"]
    total_views = sum(int(v.get("play_count", 0) or 0) for v in classified)
    yes_views = sum(int(v.get("play_count", 0) or 0) for v in yes_vids)
    no_views = sum(int(v.get("play_count", 0) or 0) for v in no_vids)
    print(f"\n=== FINAL SUMMARY ===")
    print(f"  POLITICAL (YES): {len(yes_vids)} videos, {yes_views:,} views")
    print(f"  NOT POLITICAL (NO): {len(no_vids)} videos, {no_views:,} views")


if __name__ == "__main__":
    main()
