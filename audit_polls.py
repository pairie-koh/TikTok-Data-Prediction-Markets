"""Audit TikTok polls YES entries for false positives."""
import csv, io, sys, re
from collections import Counter
csv.field_size_limit(10_000_000)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

rows = []
with open("polls/data/tiktok_polls_filtered.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

yes_rows = [r for r in rows if r.get("_topic") == "YES"]

print(f"Total YES: {len(yes_rows)}\n")

# Flag potential issues
issues = {
    "non_us": [],        # Foreign elections
    "non_political": [],  # Gallup/Pew but not political
    "no_content": [],     # Empty description + no transcript
    "gaming_polls": [],   # Game polls, TikTok polls
    "generic_vague": [],  # Very vague, might be noise
}

# Check for non-US markers
non_us_patterns = re.compile(
    r"\b(canada|canadian|uk\b|britain|british|germany|german|france|french|india|"
    r"australia|australian|brazil|mexico|nigeria|uganda|kenya|south\s+africa|"
    r"new\s+zealand|nz\b|japan|korea|israel|iran\b|iraq|turkey|pakistan|"
    r"philippines|indonesia|malaysia|thailand|colombia|argentina|peru|chile|"
    r"venezuela|honduras|guatemala|ecuador|parliament|labour\s+party|tory|"
    r"conservative\s+party|liberal\s+party|ndp\b|european\s+parliament|"
    r"lok\s+sabha|bundestag|réforme|macron|starmer|sunak|modi\b|trudeau|"
    r"bolsonaro|poilievre|orbán|orban|farage\b)",
    re.IGNORECASE
)

# Check for non-political content
non_political_patterns = re.compile(
    r"\b(mobile\s+legends|fortnite|roblox|minecraft|game\s+poll|"
    r"ai\s+usage|artificial\s+intelligence|gen\s+z\s+uses\s+ai|"
    r"mental\s+health|locus\s+of\s+control|happiness|"
    r"nfl|nba|mlb|sports|cryptocurrency|bitcoin|ethereum|"
    r"skin\s+poll|dating\s+poll|food\s+poll|movie\s+poll|"
    r"music\s+poll|fashion|beauty|skincare)\b",
    re.IGNORECASE
)

flagged = []
clean = []

for i, r in enumerate(yes_rows):
    desc = (r.get("description") or "")
    trans = (r.get("_transcript") or "")
    all_text = f"{desc} {trans}"
    kw = r.get("_search_keyword", "")
    views = int(r.get("play_count") or 0)

    flags = []

    # Non-US check
    non_us = non_us_patterns.findall(all_text)
    if non_us:
        flags.append(f"NON-US: {non_us[:3]}")

    # Non-political check
    non_pol = non_political_patterns.findall(all_text)
    if non_pol:
        flags.append(f"NON-POL: {non_pol[:3]}")

    # No content
    if len(desc.strip()) < 20 and len(trans.strip()) < 20:
        flags.append("NO_CONTENT")

    if flags:
        flagged.append((i+1, kw, views, flags, desc[:80].replace("\n"," ")))
    else:
        clean.append(r)

print(f"CLEAN: {len(clean)} / {len(yes_rows)}")
print(f"FLAGGED: {len(flagged)} / {len(yes_rows)}")
print()

# Group flagged by issue type
non_us_count = sum(1 for _,_,_,f,_ in flagged if any("NON-US" in x for x in f))
non_pol_count = sum(1 for _,_,_,f,_ in flagged if any("NON-POL" in x for x in f))
no_content_count = sum(1 for _,_,_,f,_ in flagged if any("NO_CONTENT" in x for x in f))

print(f"  NON-US flags: {non_us_count}")
print(f"  NON-POL flags: {non_pol_count}")
print(f"  NO_CONTENT flags: {no_content_count}")
print()

# Show all flagged
print("ALL FLAGGED ENTRIES:")
print("-" * 80)
for idx, kw, views, flags, desc in sorted(flagged, key=lambda x: -x[2]):
    print(f"  [{idx:>3}] Views={views:>10,} KW={kw}")
    print(f"        Flags: {', '.join(flags)}")
    print(f"        {desc}")
    print()

# View-weighted impact
flagged_views = sum(v for _,_,v,_,_ in flagged)
total_views = sum(int(r.get("play_count") or 0) for r in yes_rows)
print(f"\nVIEW IMPACT:")
print(f"  Total YES views: {total_views:,}")
print(f"  Flagged views:   {flagged_views:,} ({flagged_views/total_views*100:.1f}%)")
print(f"  Clean views:     {total_views - flagged_views:,} ({(total_views-flagged_views)/total_views*100:.1f}%)")
