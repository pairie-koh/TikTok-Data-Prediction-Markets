"""Check TikTok polls YES entries for quality."""
import csv, io, sys
csv.field_size_limit(10_000_000)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

rows = []
with open("polls/data/tiktok_polls_filtered.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

yes_rows = [r for r in rows if r.get("_topic") == "YES"]

print(f"Total YES: {len(yes_rows)}\n")

for i, r in enumerate(yes_rows):
    desc = (r.get("description") or "")[:110].replace("\n", " ")
    trans = (r.get("_transcript") or "")[:90].replace("\n", " ")
    kw = r.get("_search_keyword", "")
    mt = r.get("_match_type", "")
    views = r.get("play_count", "0")
    print(f"[{i+1:>3}] KW={kw:30s} MT={mt:15s} Views={views:>10s}")
    print(f"      {desc}")
    if trans:
        print(f"      T: {trans}")
    print()
