# Search Keywords & Hashtags

Search terms used to collect TikTok videos via Bright Data's TikTok Posts API (`gd_lu702nij2f790tmv9h`). Each keyword is submitted as a `discover_by=keyword` search. Results are deduplicated by `post_id` across all queries.

## Collection Parameters

| Parameter | Value |
|---|---|
| API | Bright Data TikTok Posts (web scraper) |
| Dataset ID | `gd_lu702nij2f790tmv9h` |
| Limit per keyword | 500 (increased from 200 in April 2026) |
| Total keywords | 71 |
| Total raw videos (deduplicated) | 5,303 |
| Collection dates | Mar-Apr 2026 |

## Keyword Categories

### Category 1: Platform + Politics (17 keywords)

High-signal queries pairing platform names with political context.

| Keyword | Videos |
|---|---|
| `polymarket election` | 27 |
| `polymarket trump` | 83 |
| `polymarket harris` | 83 |
| `polymarket biden` | 61 |
| `polymarket midterms` | 35 |
| `polymarket senate` | 35 |
| `polymarket congress` | 31 |
| `polymarket odds` | 41 |
| `polymarket vance` | 43 |
| `polymarket desantis` | 61 |
| `polymarket prediction` | 70 |
| `kalshi election` | 102 |
| `kalshi midterms` | 69 |
| `kalshi senate` | 127 |
| `kalshi odds` | 62 |
| `kalshi trump` | 79 |
| `predictit election` | 116 |

### Category 2: Generic PM + Politics (13 keywords)

| Keyword | Videos |
|---|---|
| `prediction market election` | 15 |
| `prediction markets election` | 52 |
| `prediction market odds` | 58 |
| `betting odds election` | 59 |
| `betting markets election` | 43 |
| `betting odds trump harris` | 51 |
| `betting markets trump` | 62 |
| `betting markets harris` | 92 |
| `election odds` | 49 |
| `election betting odds` | 14 |
| `election betting` | 26 |
| `midterm election odds` | 67 |
| `who will win election odds` | 69 |

### Category 3: Candidate + Odds (3 keywords)

| Keyword | Videos |
|---|---|
| `trump odds` | 84 |
| `harris odds` | 123 |
| `trump betting odds` | 35 |

### Category 4: Event-Specific (4 keywords)

| Keyword | Videos |
|---|---|
| `polymarket debate` | 39 |
| `polymarket election night` | 44 |
| `polymarket live election` | 32 |
| `polymarket results` | 19 |

### Category 5: Time-Specific (3 keywords)

| Keyword | Videos |
|---|---|
| `polymarket 2024` | 38 |
| `polymarket 2026` | 81 |
| `election odds 2026` | 110 |

### Category 6: 2026 Senate Races (4 keywords)

| Keyword | Videos |
|---|---|
| `polymarket georgia senate` | 39 |
| `polymarket michigan senate` | 79 |
| `polymarket north carolina senate` | 55 |
| `polymarket ohio senate` | 50 |

### Category 7: Hashtag Searches - Original (7 keywords)

Hashtag-prefixed searches. These consistently outperform plain keyword queries.

| Keyword | Videos |
|---|---|
| `#polymarket election` | 154 |
| `#polymarket trump` | — |
| `#kalshi election` | 158 |
| `#predictionmarkets` | 190 |
| `#electionodds` | 163 |
| `#electionbetting` | 160 |
| `#bettingodds election` | — |

### Category 8: Hashtag Searches - Expanded (20 keywords)

Added April 2026 to broaden coverage. Standalone hashtags and niche crossovers.

| Keyword | Videos |
|---|---|
| `#polymarket` | 59 |
| `#kalshi` | 95 |
| `#predictit` | 154 |
| `#bettingmarkets` | 102 |
| `#politicalbetting` | 115 |
| `#electionpredictions` | 104 |
| `#trumpodds` | 120 |
| `#election2024` | 143 |
| `#election2026` | 134 |
| `#politicalodds` | 115 |
| `#predictionmarket` | 78 |
| `#bettingodds` | 75 |
| `#electionbets` | 85 |
| `#polymarketodds` | 89 |
| `#kalshiodds` | 95 |
| `#fintok polymarket` | 0 |
| `#politicaltok polymarket` | 0 |
| `#politicaltok odds` | 0 |
| `#debateodds` | 110 |
| `#electionnight polymarket` | 0 |

### Pilot Keywords (3 keywords, not in main collection)

Used for initial pipeline validation with `limit_per_input=50`.

- `polymarket`
- `kalshi`
- `prediction market`

## Notes

- Video counts are before deduplication (same video can appear under multiple keywords)
- Compound hashtag+keyword queries (e.g., `#fintok polymarket`) returned 0 results — Bright Data's keyword search doesn't handle these well
- Hashtag-only queries consistently return more results than plain keyword queries
- 195 videos in the raw dataset have no `_search_keyword` field (from early pilot/recovery runs)
- The `_search_keyword` field on each video in `tiktok_raw.json` records which query found it
