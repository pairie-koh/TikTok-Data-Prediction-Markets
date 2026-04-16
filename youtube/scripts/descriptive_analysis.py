"""
Descriptive analysis of YouTube prediction market videos (US politics subset).

Sections 1-5: Frequency-based (no API calls)
Section 6: LLM topic & implication categorization (OpenRouter)

Usage:
    # Sections 1-5 only (free):
    python youtube/scripts/descriptive_analysis.py --no-llm

    # All sections including LLM pass:
    export OPENROUTER_API_KEY=...
    python youtube/scripts/descriptive_analysis.py
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

csv.field_size_limit(sys.maxsize)

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILTERED_CSV = DATA_DIR / "youtube_platform_filtered.csv"
LLM_PROGRESS_FILE = DATA_DIR / "descriptive_llm_progress.json"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4")

TOPIC_CATEGORIES = [
    "election_odds",
    "policy_prediction",
    "platform_regulation",
    "candidate_analysis",
    "general_political_commentary",
]

IMPLICATION_CATEGORIES = [
    "candidate_winning_or_losing",
    "market_validates_opinion",
    "market_is_wrong",
    "policy_prediction",
    "entertainment_or_spectacle",
]

CLASSIFY_PROMPT = """You are classifying a YouTube video that mentions a prediction market platform and discusses US politics.

Classify the video into exactly ONE topic and ONE implication/framing category.

TOPIC (pick one):
- election_odds: Discusses election betting odds, probabilities, or price movements
- policy_prediction: Predicts government policy outcomes (tariffs, Fed, regulation)
- platform_regulation: Discusses CFTC, Congress, or legal status of prediction markets
- candidate_analysis: Analyzes a candidate's chances, strategy, or positioning
- general_political_commentary: General political discussion that references market odds

IMPLICATION (pick one):
- candidate_winning_or_losing: Uses odds to argue someone will win or lose
- market_validates_opinion: Cites market odds as evidence their political take is correct
- market_is_wrong: Argues the market odds are mispriced or wrong
- policy_prediction: Uses markets to predict policy outcomes
- entertainment_or_spectacle: Treats prediction markets as fun, gambling, or spectacle

Respond with ONLY valid JSON on a single line, no markdown:
{{"topic": "<topic>", "implication": "<implication>"}}

---
Title: {title}
Description: {description}
Transcript: {transcript}"""


def load_politics_videos():
    videos = []
    with open(FILTERED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("_topic", "").upper() == "YES":
                videos.append(row)
    return videos


def safe_int(val, default=0):
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return default


def section_divider(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ── Section 1: Platform Distribution ──

def section1_platforms(videos):
    section_divider("SECTION 1: Platform Distribution")

    platform_counter = Counter()
    combo_counter = Counter()
    source_counter = Counter()

    for v in videos:
        platforms = v.get("_platforms", "")
        combo_counter[platforms] += 1
        for p in platforms.split(", "):
            p = p.strip()
            if p:
                platform_counter[p] += 1

        sources = v.get("_match_source", "")
        for s in sources.split(", "):
            s = s.strip()
            if s:
                source_counter[s] += 1

    n = len(videos)
    print("Platform mentions (videos can mention multiple):")
    print(f"  {'Platform':<20} {'Count':>6} {'%':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*7}")
    for p, c in platform_counter.most_common():
        print(f"  {p:<20} {c:>6} {c/n*100:>6.1f}%")

    print(f"\nPlatform combinations:")
    print(f"  {'Combination':<40} {'Count':>6} {'%':>7}")
    print(f"  {'-'*40} {'-'*6} {'-'*7}")
    for combo, c in combo_counter.most_common(15):
        print(f"  {combo:<40} {c:>6} {c/n*100:>6.1f}%")

    print(f"\nMatch source (where platform keyword was found):")
    print(f"  {'Source':<20} {'Count':>6} {'%':>7}")
    print(f"  {'-'*20} {'-'*6} {'-'*7}")
    for s, c in source_counter.most_common():
        print(f"  {s:<20} {c:>6} {c/n*100:>6.1f}%")


# ── Section 2: Engagement Stats ──

def section2_engagement(videos):
    section_divider("SECTION 2: Engagement Statistics")

    metrics = {
        "views": "Views",
        "likes": "Likes",
        "num_comments": "Comments",
    }

    for col, label in metrics.items():
        vals = [safe_int(v.get(col)) for v in videos]
        vals_nonzero = [x for x in vals if x > 0]
        total = sum(vals)
        print(f"{label}:")
        print(f"  Total:   {total:>12,}")
        if vals:
            print(f"  Mean:    {statistics.mean(vals):>12,.0f}")
            print(f"  Median:  {statistics.median(vals):>12,.0f}")
            print(f"  Min:     {min(vals):>12,}")
            print(f"  Max:     {max(vals):>12,}")
        if vals_nonzero:
            print(f"  Non-zero: {len(vals_nonzero)} / {len(vals)}")
        print()

    # Video length stats
    lengths = [safe_int(v.get("video_length")) for v in videos]
    lengths_nonzero = [x for x in lengths if x > 0]
    if lengths_nonzero:
        print("Video Length (seconds):")
        print(f"  Mean:    {statistics.mean(lengths_nonzero):>8,.0f}s ({statistics.mean(lengths_nonzero)/60:.1f} min)")
        print(f"  Median:  {statistics.median(lengths_nonzero):>8,.0f}s ({statistics.median(lengths_nonzero)/60:.1f} min)")
        print(f"  Min:     {min(lengths_nonzero):>8,}s")
        print(f"  Max:     {max(lengths_nonzero):>8,}s ({max(lengths_nonzero)/3600:.1f} hrs)")
        print()

    # Top 10 most viewed
    print("Top 10 most-viewed videos:")
    print(f"  {'#':<4} {'Views':>12}  {'Channel':<25} {'Title'}")
    print(f"  {'-'*4} {'-'*12}  {'-'*25} {'-'*50}")
    sorted_vids = sorted(videos, key=lambda v: safe_int(v.get("views")), reverse=True)
    for i, v in enumerate(sorted_vids[:10], 1):
        title = (v.get("title", "") or "")[:50].replace("\n", " ")
        channel = (v.get("youtuber", "") or "")[:25]
        views = safe_int(v.get("views"))
        print(f"  {i:<4} {views:>12,}  {channel:<25} {title}")


# ── Section 3: Temporal Distribution ──

def section3_temporal(videos):
    section_divider("SECTION 3: Temporal Distribution")

    month_counter = Counter()
    for v in videos:
        dp = v.get("date_posted", "")
        if dp:
            try:
                dt = datetime.fromisoformat(dp.replace("Z", "+00:00"))
                month_key = dt.strftime("%Y-%m")
                month_counter[month_key] += 1
            except (ValueError, TypeError):
                pass

    print(f"  {'Month':<10} {'Count':>6} {'Bar'}")
    print(f"  {'-'*10} {'-'*6} {'-'*40}")
    max_count = max(month_counter.values()) if month_counter else 1
    for month in sorted(month_counter.keys()):
        c = month_counter[month]
        bar = "#" * int(c / max_count * 40)
        print(f"  {month:<10} {c:>6} {bar}")

    print(f"\n  Total months with activity: {len(month_counter)}")


# ── Section 4: Creator Analysis ──

def section4_creators(videos):
    section_divider("SECTION 4: Creator Analysis")

    creator_counter = Counter()
    creator_views = Counter()
    for v in videos:
        channel = v.get("youtuber", "unknown")
        creator_counter[channel] += 1
        creator_views[channel] += safe_int(v.get("views"))

    print(f"Total unique channels: {len(creator_counter)}\n")

    print("Top 15 channels by video count:")
    print(f"  {'Channel':<30} {'Videos':>6} {'Total Views':>14}")
    print(f"  {'-'*30} {'-'*6} {'-'*14}")
    for ch, c in creator_counter.most_common(15):
        views = creator_views[ch]
        print(f"  {ch:<30} {c:>6} {views:>14,}")

    print(f"\nTop 15 channels by total views:")
    print(f"  {'Channel':<30} {'Total Views':>14} {'Videos':>6}")
    print(f"  {'-'*30} {'-'*14} {'-'*6}")
    for ch, views in creator_views.most_common(15):
        count = creator_counter[ch]
        print(f"  {ch:<30} {views:>14,} {count:>6}")

    # Verified breakdown
    verified = sum(1 for v in videos if v.get("verified", "").lower() == "true")
    not_verified = len(videos) - verified
    print(f"\nVerified channels: {verified} videos ({verified/len(videos)*100:.1f}%)")
    print(f"Non-verified:      {not_verified} videos ({not_verified/len(videos)*100:.1f}%)")

    # Subscriber stats
    subs = [safe_int(v.get("subscribers")) for v in videos]
    subs_nonzero = [s for s in subs if s > 0]
    if subs_nonzero:
        print(f"\nSubscriber counts (of channels in dataset):")
        print(f"  Mean:    {statistics.mean(subs_nonzero):>12,.0f}")
        print(f"  Median:  {statistics.median(subs_nonzero):>12,.0f}")
        print(f"  Max:     {max(subs_nonzero):>12,}")


# ── Section 5: Transcript Coverage ──

def section5_transcripts(videos):
    section_divider("SECTION 5: Transcript Coverage")

    has_transcript = sum(1 for v in videos if (v.get("_transcript_text", "") or "").strip())
    no_transcript = len(videos) - has_transcript

    print(f"  With transcript:    {has_transcript:>5} ({has_transcript/len(videos)*100:.1f}%)")
    print(f"  Without transcript: {no_transcript:>5} ({no_transcript/len(videos)*100:.1f}%)")

    # Transcript length stats
    lengths = [len(v.get("_transcript_text", "") or "") for v in videos if (v.get("_transcript_text", "") or "").strip()]
    if lengths:
        print(f"\n  Transcript length (characters):")
        print(f"    Mean:   {statistics.mean(lengths):>8,.0f}")
        print(f"    Median: {statistics.median(lengths):>8,.0f}")
        print(f"    Min:    {min(lengths):>8,}")
        print(f"    Max:    {max(lengths):>8,}")


# ── Section 6: LLM Topic & Implication ──

def get_llm_client():
    from openai import OpenAI
    if not OPENROUTER_API_KEY:
        print("ERROR: Set OPENROUTER_API_KEY to run LLM classification.")
        sys.exit(1)
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)


def llm_classify(client, title, description, transcript):
    prompt = CLASSIFY_PROMPT.format(
        title=title[:300],
        description=description[:500],
        transcript=transcript[:1500],
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def section6_llm(videos):
    section_divider("SECTION 6: LLM Topic & Implication Categorization")

    client = get_llm_client()
    print(f"  Model: {LLM_MODEL}")
    print(f"  Videos to classify: {len(videos)}")

    progress = {}
    if LLM_PROGRESS_FILE.exists():
        with open(LLM_PROGRESS_FILE, encoding="utf-8") as f:
            progress = json.load(f)
        print(f"  Resuming: {len(progress)} already classified\n")

    api_calls = 0
    errors = 0

    for i, v in enumerate(videos):
        vid = v.get("video_id", "")
        if vid in progress:
            continue

        title = v.get("title", "") or ""
        description = v.get("description", "") or ""
        transcript = v.get("_transcript_text", "") or ""

        try:
            result = llm_classify(client, title, description, transcript)
            topic = result.get("topic", "unknown")
            implication = result.get("implication", "unknown")

            if topic not in TOPIC_CATEGORIES:
                topic = "unknown"
            if implication not in IMPLICATION_CATEGORIES:
                implication = "unknown"

            progress[vid] = {"topic": topic, "implication": implication}
            api_calls += 1
        except Exception as e:
            print(f"    Error on {vid}: {e}")
            progress[vid] = {"topic": "error", "implication": "error"}
            errors += 1

        if (api_calls + errors) % 25 == 0:
            with open(LLM_PROGRESS_FILE, "w", encoding="utf-8") as f:
                json.dump(progress, f)
            print(f"    [{i+1}/{len(videos)}] API calls: {api_calls}, errors: {errors}")

        time.sleep(0.15)

    with open(LLM_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f)

    print(f"\n  LLM pass complete. API calls: {api_calls}, errors: {errors}\n")

    topic_counter = Counter()
    impl_counter = Counter()
    for vid_data in progress.values():
        topic_counter[vid_data["topic"]] += 1
        impl_counter[vid_data["implication"]] += 1

    n = len(progress)

    print("Topic distribution:")
    print(f"  {'Topic':<35} {'Count':>6} {'%':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*7}")
    for t, c in topic_counter.most_common():
        print(f"  {t:<35} {c:>6} {c/n*100:>6.1f}%")

    print(f"\nImplication / framing distribution:")
    print(f"  {'Implication':<35} {'Count':>6} {'%':>7}")
    print(f"  {'-'*35} {'-'*6} {'-'*7}")
    for imp, c in impl_counter.most_common():
        print(f"  {imp:<35} {c:>6} {c/n*100:>6.1f}%")

    # Cross-tab
    print(f"\nTopic x Implication cross-tab:")
    topics_sorted = [t for t, _ in topic_counter.most_common() if t not in ("unknown", "error")]
    impls_sorted = [i for i, _ in impl_counter.most_common() if i not in ("unknown", "error")]

    cross = Counter()
    for vid_data in progress.values():
        cross[(vid_data["topic"], vid_data["implication"])] += 1

    header = f"  {'':35}" + "".join(f" {imp[:15]:>15}" for imp in impls_sorted)
    print(header)
    print(f"  {'-'*35}" + "".join(f" {'-'*15}" for _ in impls_sorted))
    for t in topics_sorted:
        row = f"  {t:<35}"
        for imp in impls_sorted:
            row += f" {cross.get((t, imp), 0):>15}"
        print(row)


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Descriptive analysis of YouTube PM videos")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM classification (sections 1-5 only)")
    args = parser.parse_args()

    print("Loading US politics videos (_topic=YES)...")
    videos = load_politics_videos()
    print(f"  Loaded {len(videos)} videos\n")

    section1_platforms(videos)
    section2_engagement(videos)
    section3_temporal(videos)
    section4_creators(videos)
    section5_transcripts(videos)

    if not args.no_llm:
        section6_llm(videos)
    else:
        print("\n  [Skipping Section 6: LLM classification (--no-llm flag)]")

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
