# Prediction Markets in Political Social Media

How prediction market odds enter political discourse on TikTok and YouTube. This dataset tracks videos that reference prediction market platforms (Polymarket, Kalshi, PredictIt) and polling sources (538, RealClearPolitics, etc.) in the context of US politics.

## Dataset

| | TikTok | YouTube | Total |
|---|---|---|---|
| Prediction market videos | 593 | 4,078 | **4,671** |
| Polling-focused videos | 317 | 2,923 | **3,240** |
| **Total** | **910** | **7,001** | **7,911** |

All videos are filtered to US political content (`_topic=YES`). Date range: 2020–2026. Collection via Bright Data using 71 search keywords (see [KEYWORDS.md](KEYWORDS.md)).

### Key Findings

- **73% of TikTok** and **86% of YouTube** prediction market videos cite odds as political information, not trading advice
- "Prediction market" is overtaking "the polls" as the dominant framing in political social media content
- YouTube has significantly more down-ballot coverage (senate, house, governor) while TikTok is almost entirely presidential
- Trump is mentioned in ~60% of videos; Polymarket is the dominant platform (~70% of mentions)

## Filtering Pipeline

Each video passes a two-stage filter before entering the dataset.

**Pass 1 — Platform keyword filter** (regex, no LLM): Searches each video's description, hashtags, and transcript for platform names (`polymarket`, `kalshi`, `predictit`, `prediction market`, `betting odds`, `election odds`) with spelling variants. Only videos that explicitly name a platform or polling source survive.

**Pass 2 — US politics classifier** (Claude Sonnet via OpenRouter): Each surviving video's description and transcript are sent to Claude Sonnet, which returns YES or NO to the question: "Does this video discuss US politics specifically?" YES includes election odds, candidate predictions, policy predictions, and CFTC regulation. NO includes sports betting, foreign events, crypto tutorials, and platform promos.

The polls pipeline uses the same two-pass approach with polling-specific keywords (538, RealClearPolitics, Nate Silver, Quinnipiac, Rasmussen, etc.).

### Additional Classifications

**Information vs Trading** (`tiktok/scripts/classify_info_vs_trading.py`, `youtube/scripts/classify_info_vs_trading.py`): Claude Sonnet classifies each video's primary purpose — is the creator citing market odds as political evidence (INFORMATION) or giving trading advice (TRADING)?

**Contract Extraction** (`tiktok/scripts/extract_contracts.py`, `youtube/scripts/extract_contracts.py`): Extracts race types (presidential, senate, house, governor, policy), specific races, candidates mentioned, contracts cited, and odds mentioned from each transcript.

## Transcripts

Transcripts are sourced in priority order:
1. **Whisper transcriptions** (`tiktok/data/whisper_transcripts/`) — OpenAI Whisper, highest quality
2. **TikTok auto-captions** (`tiktok/data/tiktok_transcripts/`) — platform-generated
3. **CSV fallback** — transcript column in the filtered CSV

YouTube transcripts are embedded in `youtube/data/youtube_platform_filtered.csv` (the `_transcript_text` column).

## Repository Structure

```
final_data/                     # Clean CSVs for download
  tiktok_prediction_markets.csv   593 videos
  tiktok_polls.csv                317 videos
  youtube_prediction_markets.csv  4,078 videos
  youtube_polls.csv               2,923 videos
  transcripts_tiktok.zip          Whisper + auto-caption transcripts
  transcripts_youtube.zip

replication/                    # Self-contained replication packages
  rise_of_prediction_markets_in_political_social_media_content/
  how_creators_use_prediction_market_content/
  social_media_creators_citing_prediction_markets/
  top_20_candidates_mentioned/

trends/                         # Phrase trend analysis
  phrase_analysis.py              Script to generate all 3 charts
  markets_vs_polls_curated.png    Markets vs polls phrase usage
  phrase_trends_by_platform.png   Key phrase trends over time
  tiktok_vs_youtube_curated.png   Cross-platform vocabulary comparison

tiktok/
  data/                         # Filtered CSV, classification JSONs, transcripts
  scripts/                      # Pipeline: filter, classify, extract, transcribe

youtube/
  data/                         # Filtered CSV, classification JSONs
  scripts/                      # Pipeline: filter, classify, extract, collect

polls/
  data/                         # Filtered polls CSVs
  scripts/                      # Pipeline: filter (TikTok + YouTube)
```

## Replication

Each folder in `replication/` contains a Python script and its output PNG. All scripts read from the data files in `tiktok/data/`, `youtube/data/`, and `polls/data/`. To regenerate any chart:

```bash
python replication/<folder_name>/<folder_name>.py
```

The phrase trend analysis in `trends/` requires the same data plus transcripts:

```bash
python trends/phrase_analysis.py
```

Dependencies: `matplotlib`, `numpy`, `pandas`.
