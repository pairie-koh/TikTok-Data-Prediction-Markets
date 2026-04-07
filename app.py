"""Simple Flask app to browse TikTok prediction market transcripts."""

import csv
import json
import os
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__, static_folder="static")

DATA_DIR = Path(__file__).parent / "data"
INFO_TRADING_FILE = DATA_DIR / "info_vs_trading_progress.json"
RELEVANCE_FILE = DATA_DIR / "relevance_progress.json"


def load_json_file(path):
    """Load a JSON file if it exists."""
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def load_data():
    """Load video metadata and transcripts from the platform-filtered dataset."""
    classifications = load_json_file(INFO_TRADING_FILE)
    relevance = load_json_file(RELEVANCE_FILE)
    videos = {}

    # Load only US-politics videos (topic=YES) from platform-filtered CSV
    with open(DATA_DIR / "tiktok_platform_filtered.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("_topic", "").upper() != "YES":
                continue
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
                "prediction_market": row.get("_platforms", ""),
                "platforms": row.get("_platforms", ""),
                "match_source": row.get("_match_source", ""),
                "content_type": classifications.get(post_id, ""),
                "relevance": relevance.get(post_id, ""),
            }

    # Attach transcripts — prefer whisper, fall back to tiktok captions,
    # then fall back to the _transcript column in the filtered CSV
    whisper_dir = DATA_DIR / "whisper_transcripts"
    tiktok_dir = DATA_DIR / "tiktok_transcripts"

    # Read _transcript column from filtered CSV as fallback
    csv_transcripts = {}
    with open(DATA_DIR / "tiktok_platform_filtered.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("_transcript", "").strip()
            if t:
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
    has_transcript = request.args.get("has_transcript", "") == "true"
    market = request.args.get("market", "")
    content_type = request.args.get("content_type", "").upper()
    relevance = request.args.get("relevance", "").upper()

    # Only show relevant videos by default
    results = [v for v in VIDEO_LIST if v.get("relevance", "").upper() == "RELEVANT"]
    if has_transcript:
        results = [v for v in results if v["transcript"]]
    if market:
        results = [v for v in results if market.lower() in v.get("prediction_market", "").lower()]
    if content_type:
        results = [v for v in results if v.get("content_type", "").upper() == content_type]
    if search:
        results = [
            v for v in results
            if search in v.get("description", "").lower()
            or search in v.get("transcript", "").lower()
            or search in v.get("profile_username", "").lower()
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
                "prediction_market": v["prediction_market"],
                "content_type": v.get("content_type", ""),
                "relevance": v.get("relevance", ""),
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
