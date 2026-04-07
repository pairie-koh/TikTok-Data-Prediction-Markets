"""
YouTube Prediction Market Data Collection Pipeline.

Mirrors the TikTok pipeline (tiktok_pipeline.py) for YouTube videos.

Phases:
  1. collect         — BrightData YouTube Posts API (discover by keyword)
  2. filter-keyword  — Regex pass for platform names
  3. filter-politics — LLM: "Is this about US politics?"
  4. filter-relevance— LLM: "Is the platform mention genuine?"
  5. classify        — LLM: INFORMATION vs TRADING
  6. export          — Final CSV with all labels

Usage:
  set BRIGHTDATA_API_TOKEN=...
  set OPENROUTER_API_KEY=...
  python scripts/youtube_pipeline.py collect
  python scripts/youtube_pipeline.py filter-keyword
  python scripts/youtube_pipeline.py filter-politics
  python scripts/youtube_pipeline.py filter-relevance
  python scripts/youtube_pipeline.py classify
  python scripts/youtube_pipeline.py export
"""

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import requests

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"

RAW_JSON = DATA_RAW / "youtube_raw.json"
PLATFORM_CSV = DATA_DIR / "youtube_platform_filtered.csv"
POLITICS_PROGRESS = DATA_DIR / "youtube_filter_llm_progress.json"
RELEVANCE_PROGRESS = DATA_DIR / "youtube_relevance_progress.json"
INFO_TRADING_PROGRESS = DATA_DIR / "youtube_info_vs_trading_progress.json"
FINAL_CSV = DATA_DIR / "youtube_final.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BRIGHTDATA_API_TOKEN = os.environ.get("BRIGHTDATA_API_TOKEN", "")
BRIGHTDATA_DATASET_ID = os.environ.get(
    "BRIGHTDATA_YT_DATASET_ID", "gd_lk56epmy2i5g7lzu0k"
)  # YouTube Posts (discover by keyword)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

LIMIT_PER_INPUT = 200

# ---------------------------------------------------------------------------
# Search keywords (same as TikTok pipeline)
# ---------------------------------------------------------------------------
PLATFORM_POLITICAL = [
    "polymarket election", "polymarket trump", "polymarket harris",
    "polymarket biden", "polymarket midterms", "polymarket senate",
    "polymarket congress", "polymarket odds", "polymarket vance",
    "polymarket desantis", "polymarket prediction",
    "kalshi election", "kalshi midterms", "kalshi senate",
    "kalshi odds", "kalshi trump", "predictit election",
]

GENERIC_POLITICAL = [
    "prediction market election", "prediction markets election",
    "prediction market odds", "betting odds election",
    "betting markets election", "betting odds trump harris",
    "betting markets trump", "betting markets harris",
    "election odds", "election betting odds", "election betting",
    "midterm election odds", "who will win election odds",
]

CANDIDATE_ODDS = ["trump odds", "harris odds", "trump betting odds"]

EVENT_SPECIFIC = [
    "polymarket debate", "polymarket election night",
    "polymarket live election", "polymarket results",
]

TIME_SPECIFIC = ["polymarket 2024", "polymarket 2026", "election odds 2026"]

MIDTERM_RACES = [
    "polymarket georgia senate", "polymarket michigan senate",
    "polymarket north carolina senate", "polymarket ohio senate",
]

HASHTAG_SEARCHES = [
    "polymarket election", "polymarket trump",
    "kalshi election", "prediction markets",
    "election odds", "election betting", "betting odds election",
]

ALL_KEYWORDS = (
    PLATFORM_POLITICAL + GENERIC_POLITICAL + CANDIDATE_ODDS
    + EVENT_SPECIFIC + TIME_SPECIFIC + MIDTERM_RACES + HASHTAG_SEARCHES
)

# ---------------------------------------------------------------------------
# Platform keyword patterns
# ---------------------------------------------------------------------------
PLATFORM_PATTERNS = [
    r"polymarket", r"poly[\s\-]?market",
    r"kalshi", r"predictit", r"predict[\s\-]?it",
]
PLATFORM_RE = re.compile("|".join(PLATFORM_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# LLM helpers (same as tiktok_pipeline.py)
# ---------------------------------------------------------------------------
def _get_llm_client():
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


def _llm_call(client, model, prompt, max_tokens=10):
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


# ===================================================================
# PHASE 1: BRIGHTDATA COLLECTION
# ===================================================================

def collect_brightdata():
    if not BRIGHTDATA_API_TOKEN:
        print("ERROR: Set BRIGHTDATA_API_TOKEN environment variable")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {BRIGHTDATA_API_TOKEN}",
        "Content-Type": "application/json",
    }
    base_url = "https://api.brightdata.com/datasets/v3"

    print(f"\n  Firing {len(ALL_KEYWORDS)} triggers...")
    jobs = {}

    for i, keyword in enumerate(ALL_KEYWORDS):
        resp = requests.post(
            f"{base_url}/trigger",
            headers=headers,
            json=[{"keyword": keyword}],
            params={
                "dataset_id": BRIGHTDATA_DATASET_ID,
                "format": "json",
                "type": "discover_new",
                "discover_by": "keyword",
                "limit_per_input": LIMIT_PER_INPUT,
            },
        )
        if resp.status_code != 200:
            print(f"    [{i+1}] ERROR '{keyword}': {resp.status_code} {resp.text[:200]}")
            continue

        snapshot_id = resp.json().get("snapshot_id")
        if snapshot_id:
            jobs[snapshot_id] = keyword
            print(f"    [{i+1}/{len(ALL_KEYWORDS)}] '{keyword}' -> {snapshot_id}")
        else:
            print(f"    [{i+1}] ERROR '{keyword}': no snapshot_id")

        time.sleep(0.5)

    print(f"\n  Triggered {len(jobs)}/{len(ALL_KEYWORDS)} jobs. Polling...")

    pending = dict(jobs)
    all_results = []
    start_time = time.time()
    max_wait = 60 * 60  # 60 minutes — YouTube scraping is slow

    while pending and (time.time() - start_time) < max_wait:
        time.sleep(15)
        elapsed = int(time.time() - start_time)

        newly_done = []
        for snapshot_id, keyword in list(pending.items()):
            try:
                status_resp = requests.get(
                    f"{base_url}/progress/{snapshot_id}", headers=headers,
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
                        headers=headers, params={"format": "json"},
                    )
                    if data_resp.status_code == 200:
                        results = data_resp.json()
                        for r in results:
                            r["_search_keyword"] = keyword
                        all_results.extend(results)
                        print(f"    [{elapsed}s] '{keyword}': {records} records")
                    else:
                        print(f"    [{elapsed}s] '{keyword}': download error {data_resp.status_code}")
                else:
                    print(f"    [{elapsed}s] '{keyword}': 0 records")
            elif status == "failed":
                newly_done.append(snapshot_id)
                print(f"    [{elapsed}s] '{keyword}': FAILED")

        for sid in newly_done:
            del pending[sid]

        if newly_done:
            _save_deduped(all_results)
            done = len(jobs) - len(pending)
            print(f"    --- {done}/{len(jobs)} complete, {len(pending)} pending ---")

    if pending:
        print(f"\n  WARNING: {len(pending)} jobs timed out")

    deduped = _save_deduped(all_results)
    elapsed = int(time.time() - start_time)
    print(f"\nCollection complete in {elapsed // 60}m {elapsed % 60}s")
    print(f"  {len(deduped)} unique videos (from {len(all_results)} total)")


def _save_deduped(all_results):
    # Merge with existing data to avoid overwrites
    seen = set()
    deduped = []
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    if RAW_JSON.exists():
        try:
            existing = json.load(open(RAW_JSON, encoding="utf-8"))
            for r in existing:
                vid = r.get("video_id") or r.get("url") or r.get("id", "")
                if vid and vid not in seen:
                    seen.add(vid)
                    deduped.append(r)
        except (json.JSONDecodeError, IOError):
            pass
    for r in all_results:
        vid = r.get("video_id") or r.get("url") or r.get("id", "")
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(r)
    with open(RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    return deduped


# ===================================================================
# PHASE 2: KEYWORD FILTER
# ===================================================================

def find_platforms(text):
    matches = PLATFORM_RE.findall(text)
    platforms = set()
    for m in matches:
        ml = m.lower().replace(" ", "").replace("-", "")
        if "polymarket" in ml:
            platforms.add("polymarket")
        elif "kalshi" in ml:
            platforms.add("kalshi")
        elif "predictit" in ml:
            platforms.add("predictit")
    return sorted(platforms)


def filter_keyword():
    if not RAW_JSON.exists():
        print("ERROR: No raw data. Run 'collect' first.")
        sys.exit(1)

    with open(RAW_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} raw videos")

    # Deduplicate
    seen = set()
    deduped = []
    for v in raw:
        vid = v.get("video_id") or v.get("url") or v.get("id", "")
        if vid and vid not in seen:
            seen.add(vid)
            deduped.append(v)
    print(f"  After dedup: {len(deduped)}")

    matched = []
    for v in deduped:
        title = v.get("title", "") or ""
        description = v.get("description", "") or ""
        transcript = v.get("transcript", "") or ""
        hashtags = v.get("hashtags", "")
        if isinstance(hashtags, list):
            hashtags = " ".join(hashtags)

        found_in = []
        if find_platforms(title):
            found_in.append("title")
        if find_platforms(description):
            found_in.append("description")
        if find_platforms(transcript):
            found_in.append("transcript")
        if find_platforms(str(hashtags)):
            found_in.append("hashtags")

        all_text = f"{title} {description} {transcript} {hashtags}"
        platforms = find_platforms(all_text)

        if platforms:
            v["_platforms"] = ", ".join(platforms)
            v["_match_source"] = ", ".join(found_in)
            v["_transcript"] = transcript
            matched.append(v)

    print(f"  Pass 1 result: {len(matched)} mention a platform by name")

    plat_counts = Counter()
    for v in matched:
        for p in v["_platforms"].split(", "):
            plat_counts[p] += 1
    for p, c in plat_counts.most_common():
        print(f"    {p}: {c}")

    # Save CSV
    fieldnames = [
        "video_id", "url", "title", "description", "date_posted",
        "likes", "views", "num_comments", "video_length",
        "youtuber", "channel_url", "subscribers", "verified",
        "_search_keyword", "_platforms", "_match_source", "_topic", "_transcript",
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLATFORM_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in matched:
            row = {**v}
            if isinstance(row.get("hashtags"), list):
                row["hashtags"] = ", ".join(row["hashtags"])
            writer.writerow(row)

    print(f"  Saved {len(matched)} videos to {PLATFORM_CSV}")


# ===================================================================
# PHASE 3: LLM POLITICS FILTER
# ===================================================================

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


def filter_politics():
    if not PLATFORM_CSV.exists():
        print("ERROR: No platform-filtered data. Run 'filter-keyword' first.")
        sys.exit(1)

    with open(PLATFORM_CSV, encoding="utf-8") as f:
        videos = list(csv.DictReader(f))
    print(f"Loaded {len(videos)} platform-matched videos")

    progress = {}
    if POLITICS_PROGRESS.exists():
        with open(POLITICS_PROGRESS, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"Already classified: {len(progress)}")

    remaining = [v for v in videos if v.get("video_id", "") not in progress]
    print(f"Remaining: {len(remaining)}\n")

    if not remaining:
        counts = Counter(progress.values())
        print(f"All done. YES: {counts.get('YES', 0)}, NO: {counts.get('NO', 0)}")
        return

    client, model = _get_llm_client()
    print(f"Model: {model}")

    api_calls = 0
    for i, v in enumerate(remaining):
        vid = v.get("video_id", "")
        title = (v.get("title", "") or "")[:300]
        description = (v.get("description", "") or "")[:500]
        transcript = (v.get("_transcript", "") or "")[:1500]
        platforms = v.get("_platforms", "")

        prompt = POLITICS_PROMPT.format(
            platforms=platforms, title=title,
            description=description, transcript=transcript,
        )

        try:
            result = _llm_call(client, model, prompt, max_tokens=10).upper().strip()
            if result not in ("YES", "NO"):
                result = "YES" if "YES" in result else ("NO" if "NO" in result else "UNKNOWN")
            api_calls += 1
        except Exception as e:
            print(f"  Error on {vid}: {e}")
            result = "UNKNOWN"

        progress[vid] = result

        if (api_calls) % 25 == 0:
            with open(POLITICS_PROGRESS, "w", encoding="utf-8") as f:
                json.dump(progress, f)
            yes_n = sum(1 for v in progress.values() if v == "YES")
            no_n = sum(1 for v in progress.values() if v == "NO")
            print(f"  [{i+1}/{len(remaining)}] YES: {yes_n}, NO: {no_n}")

        time.sleep(0.15)

    with open(POLITICS_PROGRESS, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    counts = Counter(progress.values())
    n = len(progress)
    print(f"\nPolitics classification ({n} videos):")
    for cat in ["YES", "NO", "UNKNOWN"]:
        c = counts.get(cat, 0)
        if c > 0:
            print(f"  {cat}: {c} ({c / n * 100:.1f}%)")


# ===================================================================
# PHASE 4: LLM RELEVANCE FILTER
# ===================================================================

RELEVANCE_PROMPT = """A YouTube video matched our filter because it contains the keyword "{platforms}" in its {match_source}. But we need to verify: does this video genuinely discuss or reference {platforms} as a prediction market, or is the mention incidental?

RELEVANT — The video genuinely discusses {platforms}. This includes:
- Citing odds or prices from {platforms}
- Discussing events or markets on {platforms}
- Reviewing or commenting on {platforms} as a platform
- Showing {platforms} screenshots or data

NOT_RELEVANT — The mention of {platforms} is incidental. This includes:
- Keyword stuffing for visibility but the video is about something else
- Passing mention with no substantive discussion
- The video is about a completely different topic

Respond with exactly one word: RELEVANT or NOT_RELEVANT

---
Title: {title}
Description: {description}
Transcript: {transcript}"""


def filter_relevance():
    if not PLATFORM_CSV.exists():
        print("ERROR: No platform-filtered data. Run 'filter-keyword' first.")
        sys.exit(1)
    if not POLITICS_PROGRESS.exists():
        print("ERROR: No politics progress. Run 'filter-politics' first.")
        sys.exit(1)

    politics = json.load(open(POLITICS_PROGRESS, encoding="utf-8"))

    with open(PLATFORM_CSV, encoding="utf-8") as f:
        videos = [r for r in csv.DictReader(f) if politics.get(r.get("video_id", "")) == "YES"]
    print(f"Loaded {len(videos)} US politics videos")

    progress = {}
    if RELEVANCE_PROGRESS.exists():
        with open(RELEVANCE_PROGRESS, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"Already classified: {len(progress)}")

    remaining = [v for v in videos if v.get("video_id", "") not in progress]
    print(f"Remaining: {len(remaining)}\n")

    if not remaining:
        counts = Counter(progress.values())
        print(f"All done. RELEVANT: {counts.get('RELEVANT', 0)}, NOT_RELEVANT: {counts.get('NOT_RELEVANT', 0)}")
        return

    client, model = _get_llm_client()
    print(f"Model: {model}")

    api_calls = 0
    for i, v in enumerate(remaining):
        vid = v.get("video_id", "")
        title = (v.get("title", "") or "")[:300]
        description = (v.get("description", "") or "")[:500]
        transcript = (v.get("_transcript", "") or "")[:1500]
        platforms = v.get("_platforms", "")
        match_source = v.get("_match_source", "")

        prompt = RELEVANCE_PROMPT.format(
            platforms=platforms, match_source=match_source,
            title=title, description=description, transcript=transcript,
        )

        try:
            result = _llm_call(client, model, prompt, max_tokens=10).upper().strip()
            if "NOT_RELEVANT" in result or "NOT RELEVANT" in result:
                result = "NOT_RELEVANT"
            elif "RELEVANT" in result:
                result = "RELEVANT"
            else:
                result = "UNKNOWN"
            api_calls += 1
        except Exception as e:
            print(f"  Error on {vid}: {e}")
            result = "UNKNOWN"

        progress[vid] = result

        if api_calls % 25 == 0:
            with open(RELEVANCE_PROGRESS, "w", encoding="utf-8") as f:
                json.dump(progress, f)
            rel = sum(1 for v in progress.values() if v == "RELEVANT")
            notrel = sum(1 for v in progress.values() if v == "NOT_RELEVANT")
            print(f"  [{i+1}/{len(remaining)}] RELEVANT: {rel}, NOT_RELEVANT: {notrel}")

        time.sleep(0.15)

    with open(RELEVANCE_PROGRESS, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    counts = Counter(progress.values())
    n = len(progress)
    print(f"\nRelevance classification ({n} videos):")
    for cat in ["RELEVANT", "NOT_RELEVANT", "UNKNOWN"]:
        c = counts.get(cat, 0)
        if c > 0:
            print(f"  {cat}: {c} ({c / n * 100:.1f}%)")


# ===================================================================
# PHASE 5: CLASSIFY INFO vs TRADING
# ===================================================================

CLASSIFY_PROMPT = """A YouTube video mentions a prediction market platform ({platforms}). Classify the PRIMARY PURPOSE of this video.

INFORMATION — The video is about politics. The creator discusses political events, candidates, elections, or policy, and references prediction market odds as a piece of evidence or context. The point of the video is the political topic, not making money.
Examples:
- "Trump's odds on Polymarket dropped after the debate"
- "Here's what prediction markets say about the midterms"

TRADING — The video is about making money on prediction markets. The creator focuses on trading strategies, profits, portfolio advice, or how to use the platform. The political event is incidental.
Examples:
- "I made $5K on Polymarket this month, here's how"
- "Best strategy for trading election contracts on Kalshi"

Respond with exactly one word: INFORMATION or TRADING

---
Title: {title}
Description: {description}
Transcript: {transcript}"""


def classify_info_trading():
    if not PLATFORM_CSV.exists() or not POLITICS_PROGRESS.exists() or not RELEVANCE_PROGRESS.exists():
        print("ERROR: Run filter-keyword, filter-politics, and filter-relevance first.")
        sys.exit(1)

    politics = json.load(open(POLITICS_PROGRESS, encoding="utf-8"))
    relevance = json.load(open(RELEVANCE_PROGRESS, encoding="utf-8"))

    with open(PLATFORM_CSV, encoding="utf-8") as f:
        videos = [
            r for r in csv.DictReader(f)
            if politics.get(r.get("video_id", "")) == "YES"
            and relevance.get(r.get("video_id", "")) == "RELEVANT"
        ]
    print(f"Loaded {len(videos)} relevant US politics videos")

    progress = {}
    if INFO_TRADING_PROGRESS.exists():
        with open(INFO_TRADING_PROGRESS, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"Already classified: {len(progress)}")

    remaining = [v for v in videos if v.get("video_id", "") not in progress]
    print(f"Remaining: {len(remaining)}\n")

    if not remaining:
        counts = Counter(progress.values())
        print(f"All done. INFORMATION: {counts.get('INFORMATION', 0)}, TRADING: {counts.get('TRADING', 0)}")
        return

    client, model = _get_llm_client()
    print(f"Model: {model}")

    api_calls = 0
    for i, v in enumerate(remaining):
        vid = v.get("video_id", "")
        title = (v.get("title", "") or "")[:300]
        description = (v.get("description", "") or "")[:500]
        transcript = (v.get("_transcript", "") or "")[:1500]
        platforms = v.get("_platforms", "")

        prompt = CLASSIFY_PROMPT.format(
            platforms=platforms, title=title,
            description=description, transcript=transcript,
        )

        try:
            result = _llm_call(client, model, prompt, max_tokens=10).upper().strip()
            if result not in ("INFORMATION", "TRADING"):
                result = "INFORMATION" if "INFORMATION" in result else ("TRADING" if "TRADING" in result else "UNKNOWN")
            api_calls += 1
        except Exception as e:
            print(f"  Error on {vid}: {e}")
            result = "UNKNOWN"

        progress[vid] = result

        if api_calls % 25 == 0:
            with open(INFO_TRADING_PROGRESS, "w", encoding="utf-8") as f:
                json.dump(progress, f)
            info_n = sum(1 for v in progress.values() if v == "INFORMATION")
            trade_n = sum(1 for v in progress.values() if v == "TRADING")
            print(f"  [{i+1}/{len(remaining)}] INFO: {info_n}, TRADING: {trade_n}")

        time.sleep(0.15)

    with open(INFO_TRADING_PROGRESS, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    counts = Counter(progress.values())
    n = len(progress)
    print(f"\nInfo vs Trading ({n} videos):")
    for cat in ["INFORMATION", "TRADING", "UNKNOWN"]:
        c = counts.get(cat, 0)
        if c > 0:
            print(f"  {cat}: {c} ({c / n * 100:.1f}%)")


# ===================================================================
# PHASE 6: EXPORT FINAL CSV
# ===================================================================

def export_final():
    if not PLATFORM_CSV.exists():
        print("ERROR: No platform-filtered data.")
        sys.exit(1)

    politics = json.load(open(POLITICS_PROGRESS, encoding="utf-8")) if POLITICS_PROGRESS.exists() else {}
    relevance = json.load(open(RELEVANCE_PROGRESS, encoding="utf-8")) if RELEVANCE_PROGRESS.exists() else {}
    info_trading = json.load(open(INFO_TRADING_PROGRESS, encoding="utf-8")) if INFO_TRADING_PROGRESS.exists() else {}

    with open(PLATFORM_CSV, encoding="utf-8") as f:
        videos = list(csv.DictReader(f))

    final = [
        v for v in videos
        if politics.get(v.get("video_id", "")) == "YES"
        and relevance.get(v.get("video_id", "")) == "RELEVANT"
    ]

    fieldnames = [
        "video_id", "url", "title", "description", "date_posted",
        "likes", "views", "num_comments", "video_length",
        "youtuber", "channel_url", "subscribers", "verified",
        "_search_keyword", "_platforms", "_match_source",
        "_topic", "_transcript", "_relevance", "_info_vs_trading",
    ]

    with open(FINAL_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for v in final:
            vid = v.get("video_id", "")
            v["_topic"] = politics.get(vid, "")
            v["_relevance"] = relevance.get(vid, "")
            v["_info_vs_trading"] = info_trading.get(vid, "")
            writer.writerow(v)

    print(f"Exported {len(final)} videos to {FINAL_CSV}")


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="YouTube PM Data Collection Pipeline")
    parser.add_argument("phase", choices=[
        "collect", "filter-keyword", "filter-politics",
        "filter-relevance", "classify", "export",
    ])
    args = parser.parse_args()

    print("=" * 60)

    if args.phase == "collect":
        print("PHASE 1: BRIGHTDATA COLLECTION (YouTube)")
        print("=" * 60)
        collect_brightdata()

    elif args.phase == "filter-keyword":
        print("PHASE 2: KEYWORD FILTER")
        print("=" * 60)
        filter_keyword()

    elif args.phase == "filter-politics":
        print("PHASE 3: LLM POLITICS CLASSIFIER")
        print("=" * 60)
        filter_politics()

    elif args.phase == "filter-relevance":
        print("PHASE 4: LLM RELEVANCE FILTER")
        print("=" * 60)
        filter_relevance()

    elif args.phase == "classify":
        print("PHASE 5: INFO vs TRADING CLASSIFIER")
        print("=" * 60)
        classify_info_trading()

    elif args.phase == "export":
        print("PHASE 6: EXPORT FINAL CSV")
        print("=" * 60)
        export_final()


if __name__ == "__main__":
    main()
