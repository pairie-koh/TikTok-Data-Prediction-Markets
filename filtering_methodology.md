# TikTok Prediction Market Video Filtering Methodology

## Pass 1: Keyword Filter (No LLM)

The first pass is a free, regex-based keyword filter. Each video's **description**, **hashtags**, and **transcript** are searched for prediction market platform names:

- `polymarket` (including `poly market`, `poly-market`)
- `kalshi`
- `predictit` (including `predict it`, `predict-it`)

Videos that do not mention any of these platforms are dropped. Videos that match proceed to Pass 2.

---

## Pass 2: LLM Classification (Claude Sonnet)

Each surviving video is classified by an LLM (Claude Sonnet via OpenRouter) using the following prompt. The model responds with exactly one word: **YES** or **NO**.

### Prompt

```
A TikTok video mentions a prediction market platform ({platforms}). Does this video discuss US politics specifically?

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
Transcript: {transcript}
```

### Input Variables

| Variable | Source | Truncation |
|---|---|---|
| `{platforms}` | Platform names detected in Pass 1 (e.g., "kalshi", "polymarket") | None |
| `{description}` | TikTok video description text | 500 characters |
| `{transcript}` | Whisper transcription > TikTok captions > CSV fallback | 1,500 characters |

### Transcript Priority

Transcripts are loaded in order of preference:
1. **Whisper transcriptions** (OpenAI Whisper, highest quality)
2. **TikTok auto-captions** (platform-generated)
3. **CSV fallback** (previously coded transcript column)
