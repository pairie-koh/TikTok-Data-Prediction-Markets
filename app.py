"""Simple Flask app to browse TikTok prediction market transcripts."""

import csv
import os
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__, static_folder="static")

DATA_DIR = Path(__file__).parent / "data"


def load_data():
    """Load video metadata and transcripts."""
    videos = {}

    # Load from coded CSV first (has transcript column + hand-coded fields)
    with open(DATA_DIR / "tiktok_coded.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            post_id = row["post_id"]
            videos[post_id] = {
                "post_id": post_id,
                "url": row.get("url", ""),
                "description": row.get("description", ""),
                "create_time": row.get("create_time", ""),
                "profile_username": row.get("profile_username", ""),
                "digg_count": row.get("digg_count", ""),
                "share_count": row.get("share_count", ""),
                "comment_count": row.get("comment_count", ""),
                "play_count": row.get("play_count", ""),
                "video_duration": row.get("video_duration", ""),
                "hashtags": row.get("hashtags", ""),
                "prediction_market": row.get("prediction_market", ""),
                "framing": row.get("framing", ""),
                "partisan_direction": row.get("partisan_direction", ""),
                "candidate_referenced": row.get("candidate_referenced", ""),
                "sentiment": row.get("sentiment", ""),
                "coded": True,
            }

    # Load from filtered CSV (adds videos not in coded set)
    with open(DATA_DIR / "tiktok_filtered.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            post_id = row["post_id"]
            if post_id in videos:
                continue
            videos[post_id] = {
                "post_id": post_id,
                "url": row.get("url", ""),
                "description": row.get("description", ""),
                "create_time": row.get("create_time", ""),
                "profile_username": row.get("profile_username", ""),
                "digg_count": row.get("digg_count", ""),
                "share_count": row.get("share_count", ""),
                "comment_count": row.get("comment_count", ""),
                "play_count": row.get("play_count", ""),
                "video_duration": row.get("video_duration", ""),
                "hashtags": row.get("hashtags", ""),
                "prediction_market": "",
                "framing": "",
                "partisan_direction": "",
                "candidate_referenced": "",
                "sentiment": "",
                "coded": False,
            }

    # Attach transcripts — prefer whisper (better punctuation), fall back to tiktok,
    # then fall back to the transcript column in the coded CSV
    whisper_dir = DATA_DIR / "whisper_transcripts"
    tiktok_dir = DATA_DIR / "tiktok_transcripts"

    # First pass: read CSV transcript column for coded videos
    csv_transcripts = {}
    with open(DATA_DIR / "tiktok_coded.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("transcript", "").strip()
            if t:
                # Some CSV transcripts are JSON subtitle data — extract plain text
                if t.startswith("{") or t.startswith("["):
                    try:
                        import json
                        obj = json.loads(t)
                        if isinstance(obj, dict) and "utterances" in obj:
                            t = " ".join(u["text"] for u in obj["utterances"] if u.get("text"))
                        elif isinstance(obj, list):
                            t = " ".join(item.get("text", "") for item in obj if isinstance(item, dict))
                    except (json.JSONDecodeError, KeyError):
                        pass
                csv_transcripts[row["post_id"]] = t

    for post_id, video in videos.items():
        transcript = ""
        source = ""
        whisper_file = whisper_dir / f"{post_id}.txt"
        tiktok_file = tiktok_dir / f"{post_id}.txt"

        if whisper_file.exists():
            text = whisper_file.read_text(encoding="utf-8").strip()
            if text:
                transcript = text
                source = "whisper"
        if not transcript and tiktok_file.exists():
            text = tiktok_file.read_text(encoding="utf-8").strip()
            if text:
                transcript = text
                source = "tiktok"
        if not transcript and post_id in csv_transcripts:
            transcript = csv_transcripts[post_id]
            source = "csv"

        video["transcript"] = transcript
        video["transcript_source"] = source

    return videos


VIDEOS = load_data()
VIDEO_LIST = sorted(VIDEOS.values(), key=lambda v: v.get("create_time", ""), reverse=True)


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/videos")
def api_videos():
    search = request.args.get("q", "").lower()
    coded_only = request.args.get("coded", "") == "true"
    has_transcript = request.args.get("has_transcript", "") == "true"
    market = request.args.get("market", "")

    results = VIDEO_LIST
    if coded_only:
        results = [v for v in results if v["coded"]]
    if has_transcript:
        results = [v for v in results if v["transcript"]]
    if market:
        results = [v for v in results if market.lower() in v.get("prediction_market", "").lower()]
    if search:
        results = [
            v for v in results
            if search in v.get("description", "").lower()
            or search in v.get("transcript", "").lower()
            or search in v.get("profile_username", "").lower()
            or search in v.get("candidate_referenced", "").lower()
        ]

    # Return summary list (no full transcripts) for performance
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    total = len(results)
    start = (page - 1) * per_page
    page_results = results[start : start + per_page]

    return jsonify({
        "total": total,
        "page": page,
        "per_page": per_page,
        "videos": [
            {
                "post_id": v["post_id"],
                "url": v["url"],
                "description": v["description"][:200],
                "create_time": v["create_time"],
                "profile_username": v["profile_username"],
                "play_count": v["play_count"],
                "digg_count": v["digg_count"],
                "has_transcript": bool(v["transcript"]),
                "transcript_source": v["transcript_source"],
                "coded": v["coded"],
                "prediction_market": v["prediction_market"],
                "framing": v["framing"],
            }
            for v in page_results
        ],
    })


@app.route("/api/videos/<post_id>")
def api_video(post_id):
    video = VIDEOS.get(post_id)
    if not video:
        return jsonify({"error": "Not found"}), 404
    return jsonify(video)


if __name__ == "__main__":
    print(f"Loaded {len(VIDEOS)} videos ({sum(1 for v in VIDEOS.values() if v['transcript'])} with transcripts)")
    app.run(debug=True, port=5001)
