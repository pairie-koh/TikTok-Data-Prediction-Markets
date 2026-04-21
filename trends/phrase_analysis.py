"""
Phrase analysis across transcripts:
  1. Key phrases trending over time (monthly, by platform)
  2. Markets vs. Polls distinctive language
  3. TikTok vs. YouTube vocabulary comparison
"""

import csv, json, re, os, sys
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

csv.field_size_limit(10_000_000)
ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────
BLUE   = "#3B82F6"
RED    = "#EF4444"
GREEN  = "#10B981"
ORANGE = "#F59E0B"
PURPLE = "#8B5CF6"
PINK   = "#EC4899"
TEAL   = "#14B8A6"
GREY   = "#6B7280"

# ── 1. Load data ────────────────────────────────────────────────────

def load_tiktok_pm():
    """TikTok prediction-market videos (topic=YES, has transcript)."""
    path = ROOT / "final_data" / "tiktok_prediction_markets.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("_topic") == "YES" and r.get("_transcript", "").strip():
                rows.append({
                    "id": r["post_id"],
                    "date": r["create_time"][:7],       # YYYY-MM
                    "transcript": r["_transcript"],
                    "platform": "TikTok",
                    "content_type": "markets",
                })
    return rows


def load_tiktok_polls():
    """TikTok polls-focused videos (topic=YES, has transcript)."""
    # exclude ids already in PM set
    pm_ids = {r["id"] for r in load_tiktok_pm()}
    path = ROOT / "final_data" / "tiktok_polls.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("_topic") == "YES" and r.get("_transcript", "").strip():
                if r["post_id"] not in pm_ids:
                    rows.append({
                        "id": r["post_id"],
                        "date": r["create_time"][:7],
                        "transcript": r["_transcript"],
                        "platform": "TikTok",
                        "content_type": "polls",
                    })
    return rows


def load_youtube_pm():
    """YouTube prediction-market videos (topic=YES, has transcript)."""
    path = ROOT / "final_data" / "youtube_prediction_markets.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("_topic") == "YES" and r.get("_transcript_text", "").strip():
                rows.append({
                    "id": r["video_id"],
                    "date": r.get("date_posted", "")[:7],
                    "transcript": r["_transcript_text"],
                    "platform": "YouTube",
                    "content_type": "markets",
                })
    return rows


def load_youtube_polls():
    """YouTube polls-focused videos (topic=YES, has transcript)."""
    pm_ids = {r["id"] for r in load_youtube_pm()}
    path = ROOT / "final_data" / "youtube_polls.csv"
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("_topic") == "YES" and r.get("_transcript_text", "").strip():
                if r.get("video_id","") not in pm_ids:
                    rows.append({
                        "id": r.get("video_id",""),
                        "date": r.get("date_posted", "")[:7],
                        "transcript": r["_transcript_text"],
                        "platform": "YouTube",
                        "content_type": "polls",
                    })
    return rows


print("Loading data …")
tk_pm    = load_tiktok_pm()
tk_polls = load_tiktok_polls()
# Cache tiktok PM so polls loader can use it
_tk_pm_cache = tk_pm
print(f"  TikTok PM:     {len(tk_pm):,}")
print(f"  TikTok Polls:  {len(tk_polls):,}")

yt_pm    = load_youtube_pm()
yt_polls = load_youtube_polls()
print(f"  YouTube PM:    {len(yt_pm):,}")
print(f"  YouTube Polls: {len(yt_polls):,}")

ALL = tk_pm + tk_polls + yt_pm + yt_polls
print(f"  Total:         {len(ALL):,}")


# ── helpers ─────────────────────────────────────────────────────────

def phrase_rate(docs, phrase):
    """Fraction of docs whose transcript contains phrase (case-insensitive)."""
    p = phrase.lower()
    n = sum(1 for d in docs if p in d["transcript"].lower())
    return n / len(docs) if docs else 0


def phrase_rate_by_month(docs, phrase):
    """Returns {month: rate} for months with ≥5 docs."""
    by_month = defaultdict(list)
    for d in docs:
        by_month[d["date"]].append(d)
    result = {}
    for m in sorted(by_month):
        if len(by_month[m]) >= 5:
            result[m] = phrase_rate(by_month[m], phrase)
    return result


def log_odds_ratio(docs_a, docs_b, min_count=3):
    """
    Informative Dirichlet prior log-odds ratio between two corpora.
    Returns list of (word, log_odds, z_score) sorted by |z_score|.
    """
    def word_counts(docs):
        c = Counter()
        for d in docs:
            words = set(re.findall(r"[a-z']+", d["transcript"].lower()))
            c.update(words)
        return c

    ca, cb = word_counts(docs_a), word_counts(docs_b)
    na, nb = len(docs_a), len(docs_b)
    vocab = set(k for k, v in ca.items() if v >= min_count) | \
            set(k for k, v in cb.items() if v >= min_count)

    # uninformative prior α = 1
    alpha = 1.0
    V = len(vocab)
    results = []
    for w in vocab:
        ya = ca[w] + alpha
        yb = cb[w] + alpha
        sa = na + V * alpha
        sb = nb + V * alpha
        log_odds = np.log(ya / sa) - np.log(yb / sb)
        variance = 1.0 / ya + 1.0 / yb
        z = log_odds / np.sqrt(variance)
        results.append((w, log_odds, z))
    results.sort(key=lambda x: abs(x[2]), reverse=True)
    return results


def bigram_counts(docs):
    """Count bigrams (as 'word1 word2') across doc set."""
    c = Counter()
    for d in docs:
        words = re.findall(r"[a-z']+", d["transcript"].lower())
        for a, b in zip(words, words[1:]):
            c[f"{a} {b}"] += 1
    return c


def bigram_log_odds(docs_a, docs_b, min_count=5):
    """Log-odds for bigrams."""
    ca, cb = bigram_counts(docs_a), bigram_counts(docs_b)
    na, nb = sum(ca.values()), sum(cb.values())
    vocab = set(k for k, v in ca.items() if v >= min_count) | \
            set(k for k, v in cb.items() if v >= min_count)
    alpha = 1.0
    V = len(vocab)
    results = []
    for w in vocab:
        ya = ca[w] + alpha
        yb = cb[w] + alpha
        sa = na + V * alpha
        sb = nb + V * alpha
        log_odds = np.log(ya / sa) - np.log(yb / sb)
        variance = 1.0 / ya + 1.0 / yb
        z = log_odds / np.sqrt(variance)
        results.append((w, log_odds, z))
    results.sort(key=lambda x: abs(x[2]), reverse=True)
    return results


# ── stop words for log-odds filtering ──────────────────────────────
STOP = {
    "the", "a", "an", "to", "of", "in", "is", "and", "that", "it", "for",
    "on", "with", "this", "was", "are", "be", "at", "or", "as", "but", "not",
    "by", "from", "they", "we", "he", "she", "you", "i", "my", "his", "her",
    "our", "their", "its", "have", "has", "had", "do", "does", "did", "will",
    "would", "can", "could", "should", "may", "might", "been", "being",
    "just", "so", "if", "than", "then", "about", "up", "out", "no", "what",
    "all", "when", "how", "who", "which", "there", "them", "me", "him",
    "us", "your", "these", "those", "such", "very", "too", "also", "more",
    "some", "any", "each", "every", "other", "many", "much", "most", "own",
    "same", "both", "few", "s", "t", "re", "ve", "ll", "d", "m", "don",
    "doesn", "didn", "wasn", "weren", "isn", "aren", "haven", "hasn",
    "won", "wouldn", "couldn", "shouldn", "let", "get", "got", "going",
    "go", "went", "come", "came", "make", "made", "take", "took", "know",
    "knew", "think", "thought", "say", "said", "tell", "told", "see",
    "saw", "give", "gave", "look", "looked", "want", "wanted", "use",
    "used", "way", "thing", "things", "like", "right", "now", "still",
    "well", "back", "even", "new", "good", "first", "last", "long",
    "great", "little", "own", "old", "big", "high", "different", "small",
    "large", "next", "early", "young", "important", "here", "why", "while",
    "where", "after", "before", "over", "between", "under", "again",
    "further", "once", "during", "off", "down", "only", "into", "through",
    "able", "around", "lot", "really", "actually", "kind", "guys",
    "yeah", "okay", "oh", "um", "uh", "gonna", "gotta", "wanna",
    "one", "two", "three", "four", "five", "ten", "hundred", "thousand",
    "million", "billion", "year", "years", "day", "days", "time", "people",
    "percent", "number", "part", "put", "keep", "mean", "end", "point",
    "world", "country",
}


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 – Key phrase trends over time
# ═══════════════════════════════════════════════════════════════════

print("\n── Figure 1: Phrase trends over time ──")

# Curated phrase groups that tell a story
PHRASE_GROUPS = {
    "How creators reference odds sources": {
        "phrases": {
            "the polls":          {"color": BLUE,   "ls": "-"},
            "the odds":           {"color": RED,    "ls": "-"},
            "prediction market":  {"color": GREEN,  "ls": "-"},
            "polymarket":         {"color": ORANGE, "ls": "-"},
            "kalshi":             {"color": PURPLE, "ls": "-"},
            "betting odds":       {"color": PINK,   "ls": "--"},
        },
        "ylabel": "% of videos mentioning phrase",
    },
    "Authority framing: how odds are cited": {
        "phrases": {
            "according to":     {"color": BLUE,   "ls": "-"},
            "chances of":       {"color": RED,    "ls": "-"},
            "% chance":         {"color": GREEN,  "ls": "-"},
            "probability":      {"color": ORANGE, "ls": "-"},
            "the market says":  {"color": PURPLE, "ls": "--"},
            "markets are":      {"color": PINK,   "ls": "--"},
        },
        "ylabel": "% of videos mentioning phrase",
    },
    "Framing: gambling vs. investing vs. information": {
        "phrases": {
            "gambling":    {"color": RED,    "ls": "-"},
            "bet on":      {"color": ORANGE, "ls": "-"},
            "make money":  {"color": GREEN,  "ls": "-"},
            "legal":       {"color": BLUE,   "ls": "-"},
            "regulate":    {"color": PURPLE, "ls": "-"},
        },
        "ylabel": "% of videos mentioning phrase",
    },
}

# Use all markets docs (both platforms) for time trends
all_markets = [d for d in ALL if d["content_type"] == "markets"]

fig, axes = plt.subplots(len(PHRASE_GROUPS), 1, figsize=(14, 5 * len(PHRASE_GROUPS)),
                         sharex=False)
if len(PHRASE_GROUPS) == 1:
    axes = [axes]

for ax, (title, cfg) in zip(axes, PHRASE_GROUPS.items()):
    for phrase, style in cfg["phrases"].items():
        rates = phrase_rate_by_month(all_markets, phrase)
        if not rates:
            continue
        months = sorted(rates.keys())
        vals = [rates[m] * 100 for m in months]

        # 3-month rolling average for smoothing
        if len(vals) >= 3:
            smoothed = []
            for i in range(len(vals)):
                window = vals[max(0, i-1):i+2]
                smoothed.append(sum(window) / len(window))
        else:
            smoothed = vals

        ax.plot(range(len(months)), smoothed,
                color=style["color"], linestyle=style["ls"],
                linewidth=2.2, label=f'"{phrase}"', marker="o", markersize=3)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(cfg["ylabel"], fontsize=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # x-axis tick labels
    months = sorted(set(d["date"] for d in all_markets if len(d["date"]) == 7))
    months = [m for m in months if sum(1 for d in all_markets if d["date"] == m) >= 5]
    tick_pos = list(range(len(months)))
    # Show every 3rd label
    labels = [m if i % 3 == 0 else "" for i, m in enumerate(months)]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

plt.suptitle("Phrase Trends in Prediction Market Videos Over Time\n(TikTok + YouTube, 3-month rolling avg, months with ≥5 videos)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "phrase_trends_over_time.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT / 'phrase_trends_over_time.png'}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 – Markets vs. Polls: distinctive language
# ═══════════════════════════════════════════════════════════════════

print("\n── Figure 2: Markets vs. Polls distinctive language ──")

markets_docs = [d for d in ALL if d["content_type"] == "markets"]
polls_docs   = [d for d in ALL if d["content_type"] == "polls"]

print(f"  Markets corpus: {len(markets_docs):,} docs")
print(f"  Polls corpus:   {len(polls_docs):,} docs")

# --- 2a: unigram log-odds ---
unigram_results = log_odds_ratio(markets_docs, polls_docs, min_count=5)
# Filter stop words
unigram_filtered = [(w, lo, z) for w, lo, z in unigram_results if w not in STOP]

# Top 20 markets-leaning, top 20 polls-leaning
markets_top = [(w, z) for w, lo, z in unigram_filtered if z > 0][:20]
polls_top   = [(w, z) for w, lo, z in unigram_filtered if z < 0][:20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Markets words
words_m = [w for w, z in reversed(markets_top)]
scores_m = [z for w, z in reversed(markets_top)]
bars1 = ax1.barh(words_m, scores_m, color=BLUE, alpha=0.85)
ax1.set_xlabel("z-score (higher = more distinctive to Markets)", fontsize=10)
ax1.set_title("Words distinctive to\nPrediction Market videos", fontsize=12, fontweight="bold")
ax1.grid(True, axis="x", alpha=0.3)

# Polls words
words_p = [w for w, z in reversed(polls_top)]
scores_p = [abs(z) for w, z in reversed(polls_top)]
bars2 = ax2.barh(words_p, scores_p, color=RED, alpha=0.85)
ax2.set_xlabel("z-score (higher = more distinctive to Polls)", fontsize=10)
ax2.set_title("Words distinctive to\nPolling-focused videos", fontsize=12, fontweight="bold")
ax2.grid(True, axis="x", alpha=0.3)

plt.suptitle("Markets vs. Polls: Most Distinctive Vocabulary\n(Log-odds ratio with informative Dirichlet prior, unigrams)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "markets_vs_polls_unigrams.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT / 'markets_vs_polls_unigrams.png'}")

# --- 2b: bigram log-odds ---
bigram_results = bigram_log_odds(markets_docs, polls_docs, min_count=5)

# Filter bigrams where both words are stop words
bigram_filtered = []
for bg, lo, z in bigram_results:
    parts = bg.split()
    if parts[0] in STOP and parts[1] in STOP:
        continue
    bigram_filtered.append((bg, lo, z))

markets_bi = [(w, z) for w, lo, z in bigram_filtered if z > 0][:20]
polls_bi   = [(w, z) for w, lo, z in bigram_filtered if z < 0][:20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

words_m = [w for w, z in reversed(markets_bi)]
scores_m = [z for w, z in reversed(markets_bi)]
ax1.barh(words_m, scores_m, color=BLUE, alpha=0.85)
ax1.set_xlabel("z-score", fontsize=10)
ax1.set_title("Bigrams distinctive to\nPrediction Market videos", fontsize=12, fontweight="bold")
ax1.grid(True, axis="x", alpha=0.3)

words_p = [w for w, z in reversed(polls_bi)]
scores_p = [abs(z) for w, z in reversed(polls_bi)]
ax2.barh(words_p, scores_p, color=RED, alpha=0.85)
ax2.set_xlabel("z-score", fontsize=10)
ax2.set_title("Bigrams distinctive to\nPolling-focused videos", fontsize=12, fontweight="bold")
ax2.grid(True, axis="x", alpha=0.3)

plt.suptitle("Markets vs. Polls: Most Distinctive Bigrams\n(Log-odds ratio with informative Dirichlet prior)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "markets_vs_polls_bigrams.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT / 'markets_vs_polls_bigrams.png'}")


# --- 2c: Curated phrase comparison (grouped bar) ---
CURATED_PHRASES = [
    # How odds are referenced
    "the polls", "the odds", "prediction market", "betting odds",
    "according to", "chances of", "% chance", "probability",
    # Specific sources
    "polymarket", "kalshi", "538", "real clear",
    "nate silver", "rasmussen",
    # Framing
    "margin of error", "sample size", "likely voters",
    "gambling", "bet on", "make money",
    "swing state", "battleground",
    # Candidates
    "trump", "harris", "biden",
]

fig, ax = plt.subplots(figsize=(14, 10))
y_pos = np.arange(len(CURATED_PHRASES))
width = 0.35

markets_rates = [phrase_rate(markets_docs, p) * 100 for p in CURATED_PHRASES]
polls_rates   = [phrase_rate(polls_docs, p) * 100 for p in CURATED_PHRASES]

bars1 = ax.barh(y_pos + width/2, markets_rates, width, label="Prediction Markets",
                color=BLUE, alpha=0.85)
bars2 = ax.barh(y_pos - width/2, polls_rates, width, label="Polls",
                color=RED, alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'"{p}"' for p in CURATED_PHRASES], fontsize=9)
ax.set_xlabel("% of videos containing phrase", fontsize=11)
ax.set_title("Prediction Market vs. Polling Videos: Phrase Usage\n(All platforms combined)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, axis="x", alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUT / "markets_vs_polls_curated.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT / 'markets_vs_polls_curated.png'}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 – TikTok vs. YouTube vocabulary
# ═══════════════════════════════════════════════════════════════════

print("\n── Figure 3: TikTok vs. YouTube vocabulary ──")

tiktok_docs  = [d for d in ALL if d["platform"] == "TikTok"]
youtube_docs = [d for d in ALL if d["platform"] == "YouTube"]
print(f"  TikTok corpus:  {len(tiktok_docs):,} docs")
print(f"  YouTube corpus: {len(youtube_docs):,} docs")

# --- 3a: unigram log-odds ---
plat_unigrams = log_odds_ratio(tiktok_docs, youtube_docs, min_count=5)
plat_filtered = [(w, lo, z) for w, lo, z in plat_unigrams if w not in STOP]

tk_top = [(w, z) for w, lo, z in plat_filtered if z > 0][:20]
yt_top = [(w, z) for w, lo, z in plat_filtered if z < 0][:20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

words_t = [w for w, z in reversed(tk_top)]
scores_t = [z for w, z in reversed(tk_top)]
ax1.barh(words_t, scores_t, color=PINK, alpha=0.85)
ax1.set_xlabel("z-score (higher = more distinctive to TikTok)", fontsize=10)
ax1.set_title("Words distinctive to TikTok", fontsize=12, fontweight="bold")
ax1.grid(True, axis="x", alpha=0.3)

words_y = [w for w, z in reversed(yt_top)]
scores_y = [abs(z) for w, z in reversed(yt_top)]
ax2.barh(words_y, scores_y, color=RED, alpha=0.85)
ax2.set_xlabel("z-score (higher = more distinctive to YouTube)", fontsize=10)
ax2.set_title("Words distinctive to YouTube", fontsize=12, fontweight="bold")
ax2.grid(True, axis="x", alpha=0.3)

plt.suptitle("TikTok vs. YouTube: Most Distinctive Vocabulary\n(Log-odds ratio with informative Dirichlet prior)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "tiktok_vs_youtube_unigrams.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT / 'tiktok_vs_youtube_unigrams.png'}")

# --- 3b: bigram log-odds ---
plat_bigrams = bigram_log_odds(tiktok_docs, youtube_docs, min_count=5)
plat_bi_filtered = []
for bg, lo, z in plat_bigrams:
    parts = bg.split()
    if parts[0] in STOP and parts[1] in STOP:
        continue
    plat_bi_filtered.append((bg, lo, z))

tk_bi = [(w, z) for w, lo, z in plat_bi_filtered if z > 0][:20]
yt_bi = [(w, z) for w, lo, z in plat_bi_filtered if z < 0][:20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))

words_t = [w for w, z in reversed(tk_bi)]
scores_t = [z for w, z in reversed(tk_bi)]
ax1.barh(words_t, scores_t, color=PINK, alpha=0.85)
ax1.set_xlabel("z-score", fontsize=10)
ax1.set_title("Bigrams distinctive to TikTok", fontsize=12, fontweight="bold")
ax1.grid(True, axis="x", alpha=0.3)

words_y = [w for w, z in reversed(yt_bi)]
scores_y = [abs(z) for w, z in reversed(yt_bi)]
ax2.barh(words_y, scores_y, color=RED, alpha=0.85)
ax2.set_xlabel("z-score", fontsize=10)
ax2.set_title("Bigrams distinctive to YouTube", fontsize=12, fontweight="bold")
ax2.grid(True, axis="x", alpha=0.3)

plt.suptitle("TikTok vs. YouTube: Most Distinctive Bigrams\n(Log-odds ratio with informative Dirichlet prior)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "tiktok_vs_youtube_bigrams.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT / 'tiktok_vs_youtube_bigrams.png'}")


# --- 3c: curated phrase comparison across platforms ---
PLATFORM_PHRASES = [
    # Platform mentions
    "polymarket", "kalshi", "predictit",
    # Formality/authority
    "according to", "the data shows", "probability",
    "% chance", "the odds",
    # Colloquial/engagement
    "bet on", "put money on", "gambling",
    "make money", "cash out",
    # Electoral
    "swing state", "battleground", "electoral college",
    "senate", "house", "governor",
    # Regulation
    "cftc", "regulate", "legal",
    # Candidates
    "trump", "harris", "biden", "vance",
]

fig, ax = plt.subplots(figsize=(14, 10))
y_pos = np.arange(len(PLATFORM_PHRASES))
width = 0.35

tk_rates = [phrase_rate(tiktok_docs, p) * 100 for p in PLATFORM_PHRASES]
yt_rates = [phrase_rate(youtube_docs, p) * 100 for p in PLATFORM_PHRASES]

bars1 = ax.barh(y_pos + width/2, tk_rates, width, label="TikTok",
                color=PINK, alpha=0.85)
bars2 = ax.barh(y_pos - width/2, yt_rates, width, label="YouTube",
                color=RED, alpha=0.85)

ax.set_yticks(y_pos)
ax.set_yticklabels([f'"{p}"' for p in PLATFORM_PHRASES], fontsize=9)
ax.set_xlabel("% of videos containing phrase", fontsize=11)
ax.set_title("TikTok vs. YouTube: Phrase Usage Comparison\n(All content types combined)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, axis="x", alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUT / "tiktok_vs_youtube_curated.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT / 'tiktok_vs_youtube_curated.png'}")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 – Phrase trends split by platform (TikTok vs YouTube)
# ═══════════════════════════════════════════════════════════════════

print("\n── Figure 4: Phrase trends split by platform ──")

KEY_PHRASES = [
    ("polymarket", ORANGE),
    ("prediction market", GREEN),
    ("the polls", BLUE),
    ("the odds", RED),
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=False)

for ax, platform, docs, title in [
    (ax1, "TikTok",  [d for d in ALL if d["platform"] == "TikTok" and d["content_type"] == "markets"], "TikTok – Prediction Market Videos"),
    (ax2, "YouTube", [d for d in ALL if d["platform"] == "YouTube" and d["content_type"] == "markets"], "YouTube – Prediction Market Videos"),
]:
    for phrase, color in KEY_PHRASES:
        rates = phrase_rate_by_month(docs, phrase)
        if not rates:
            continue
        months = sorted(rates.keys())
        vals = [rates[m] * 100 for m in months]
        # smooth
        if len(vals) >= 3:
            smoothed = [sum(vals[max(0,i-1):i+2])/len(vals[max(0,i-1):i+2]) for i in range(len(vals))]
        else:
            smoothed = vals
        ax.plot(range(len(months)), smoothed, color=color, linewidth=2,
                label=f'"{phrase}"', marker="o", markersize=3)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("% of videos", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # x labels
    months = sorted(set(d["date"] for d in docs if len(d["date"]) == 7))
    months = [m for m in months if sum(1 for d in docs if d["date"] == m) >= 5]
    labels = [m if i % 3 == 0 else "" for i, m in enumerate(months)]
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

plt.suptitle("Key Phrase Trends: TikTok vs. YouTube\n(3-month rolling avg, months with ≥5 videos)",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUT / "phrase_trends_by_platform.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT / 'phrase_trends_by_platform.png'}")


print("\n✓ All figures saved to", OUT)
