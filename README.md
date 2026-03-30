# TikTok-Data-Prediction-Markets

TikTok videos discussing prediction markets (Polymarket, Kalshi, PredictIt) in the context of US politics. Built for academic research on how prediction market odds enter political discourse on social media.

## Dataset Summary

| Metric | Value |
|---|---|
| Raw scraped videos | ~2,800+ |
| After platform keyword filter | 906 |
| After US politics classifier | 413 |
| After relevance filter (final) | **364** |
| Date range | Nov 2020 - Mar 2026 |
| Total views (364 videos) | ~27M |

### Platform Breakdown (364 relevant videos)

- Polymarket: ~70%
- Kalshi: ~41%
- PredictIt: 0.5%
- Both Kalshi + Polymarket: ~12%

### Content Classification (413 US politics videos)

- **INFORMATION (68%):** Creator discusses politics and cites prediction market odds as evidence or context
- **TRADING (32%):** Creator focuses on making money on prediction markets; the political event is incidental

## Filtering Pipeline

The raw scrape was filtered in three passes to arrive at the final 364-video dataset.

### Filter 1: Platform Keyword Match (`scripts/filter_by_platform.py`, Pass 1)

Regex-based keyword filter (no LLM, free). Each video's description, hashtags, and transcript are searched for platform names: `polymarket`, `kalshi`, `predictit` (with spelling variants). Videos with no platform mention are dropped.

- **Input:** ~2,800+ raw videos
- **Output:** 906 videos

### Filter 2: US Politics Classifier (`scripts/filter_by_platform.py`, Pass 2)

LLM classifier (Claude Sonnet via OpenRouter) asks: "Does this video discuss US politics specifically?" Responds YES or NO. YES includes election odds, candidate predictions, policy predictions, CFTC regulation discussion. NO includes sports betting, foreign events, crypto content, platform tutorials.

- **Input:** 906 platform-matched videos
- **Output:** 413 YES (US politics), 491 NO

### Filter 3: Relevance Filter (`scripts/filter_relevance.py`)

LLM classifier (Claude Sonnet via OpenRouter) checks whether the platform mention is genuine or incidental. Filters out hashtag spam, ads tagging a competitor keyword, and passing mentions with no substantive discussion.

- **Input:** 413 US politics videos
- **Output:** 364 RELEVANT, 49 NOT_RELEVANT

## Additional Classifications

### Information vs Trading (`scripts/classify_info_vs_trading.py`)

Classifies each video's primary purpose:
- **INFORMATION:** Video is about politics; prediction market odds are cited as evidence
- **TRADING:** Video is about making money; the political event is incidental

### Framing and Staleness (`scripts/classify_framing_and_staleness.py`)

Two classifications in one pass:
1. **Framing:** Does the creator present odds as neutral data or partisan ammunition?
2. **Staleness:** Does the video cite specific odds that would become outdated, or is it general commentary?

### Descriptive Analysis (`scripts/descriptive_analysis.py`)

Frequency-based descriptive statistics (platform breakdown, candidate mentions, engagement, temporal distribution) with an optional LLM pass for topic categorization.

## Transcripts

Transcripts are sourced in priority order:
1. **Whisper transcriptions** (`data/whisper_transcripts/`) - OpenAI Whisper, highest quality
2. **TikTok auto-captions** (`data/tiktok_transcripts/`) - platform-generated
3. **CSV fallback** - previously coded transcript column

## Data Files

| File | Description |
|---|---|
| `data/raw/tiktok_raw.json` | Raw scraped data from BrightData |
| `data/tiktok_platform_filtered.csv` | 906 videos after keyword + US politics filter |
| `data/filter_llm_progress.json` | Pass 2 (US politics) classification progress |
| `data/relevance_progress.json` | Filter 3 (relevance) classification progress |
| `data/info_vs_trading_progress.json` | Information vs Trading classification progress |
| `data/framing_staleness_progress.json` | Framing & staleness classification progress |

## Notes

- The dataset has some degree of **recency bias** because TikTok's search algorithm favors recent content during scraping
- There was a large spike in video volume around **Oct-Nov 2024** (US presidential election)
- Legacy media accounts (CBS Mornings, Daily Mail, 60 Minutes) dominate the top-viewed videos
- Trump is mentioned in ~60% of videos, Harris in ~13-19%, Biden in ~8%, Vance in ~7%