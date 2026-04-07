"""
Whisper transcription for TikTok videos using the OpenAI API.

Downloads audio via yt-dlp, sends to OpenAI Whisper API, saves transcripts.
Runs with parallel workers to speed things up.

Usage:
  python scripts/whisper_transcribe.py                    # transcribe all missing
  python scripts/whisper_transcribe.py --workers 10       # 10 parallel workers
  python scripts/whisper_transcribe.py --test 5           # test on 5 videos
  python scripts/whisper_transcribe.py --stats             # just print stats
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "tiktok_raw.json"
WHISPER_DIR = DATA_DIR / "whisper_transcripts"
WHISPER_DIR.mkdir(parents=True, exist_ok=True)

# Track failed video IDs so we don't retry them endlessly
FAILED_LOG = DATA_DIR / "whisper_failed.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def load_raw_videos():
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_missing_ids(videos):
    """Return list of (post_id, url) for videos without whisper transcripts."""
    # Load previously failed IDs to skip
    failed_ids = set()
    if FAILED_LOG.exists():
        with open(FAILED_LOG, "r", encoding="utf-8") as f:
            failed_ids = set(json.load(f))

    missing = []
    for v in videos:
        pid = v.get("post_id", "")
        url = v.get("url", "")
        if not pid or not url:
            continue
        transcript_path = WHISPER_DIR / f"{pid}.txt"
        if transcript_path.exists():
            continue
        if pid in failed_ids:
            continue
        missing.append((pid, url))
    return missing


def download_video(url, output_path, timeout=60):
    """Download video from a TikTok URL using yt-dlp (mp4, no ffmpeg needed)."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "worst",  # smallest file = faster download + upload
        "-o", str(output_path),
        "--no-warnings",
        "--quiet",
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode == 0


def transcribe_audio(client, audio_path):
    """Send audio file to OpenAI Whisper API and return transcript text."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    return response.strip()


def process_one(pid, url, client):
    """Download + transcribe a single video. Returns (pid, success, error_msg)."""
    transcript_path = WHISPER_DIR / f"{pid}.txt"

    # Use a temp dir for the video download
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = Path(tmpdir) / f"{pid}.mp4"

        # Step 1: Download video (smallest quality, no ffmpeg needed)
        try:
            ok = download_video(url, video_path)
            if not ok or not video_path.exists():
                # yt-dlp may pick a different extension
                candidates = list(Path(tmpdir).glob(f"{pid}.*"))
                if candidates:
                    video_path = candidates[0]
                else:
                    return (pid, False, "download_failed")
        except subprocess.TimeoutExpired:
            return (pid, False, "download_timeout")
        except Exception as e:
            return (pid, False, f"download_error: {e}")

        # Check file size (skip if too large for API - 25MB limit)
        if video_path.stat().st_size > 24 * 1024 * 1024:
            return (pid, False, "file_too_large")

        if video_path.stat().st_size < 1000:
            return (pid, False, "file_too_small")

        # Step 2: Transcribe via Whisper API (accepts mp4 directly)
        try:
            text = transcribe_audio(client, video_path)
            if text:
                transcript_path.write_text(text, encoding="utf-8")
                return (pid, True, None)
            else:
                return (pid, False, "empty_transcript")
        except Exception as e:
            return (pid, False, f"whisper_error: {e}")


def run_transcription(workers=5, max_videos=None):
    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client = OpenAI(api_key=OPENAI_API_KEY)

    print("Loading raw videos...")
    videos = load_raw_videos()
    missing = get_missing_ids(videos)

    if max_videos:
        missing = missing[:max_videos]

    if not missing:
        print("All videos already have whisper transcripts!")
        return

    print(f"Videos to transcribe: {len(missing)}")
    print(f"Workers: {workers}")
    print()

    success_count = 0
    fail_count = 0
    failed_ids = []

    # Load existing failed log
    if FAILED_LOG.exists():
        with open(FAILED_LOG, "r", encoding="utf-8") as f:
            failed_ids = json.load(f)

    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for pid, url in missing:
            future = executor.submit(process_one, pid, url, client)
            futures[future] = pid

        for i, future in enumerate(as_completed(futures), 1):
            pid, ok, err = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                failed_ids.append(pid)
                if i <= 20 or i % 100 == 0:
                    print(f"  FAIL [{pid}]: {err}")

            if i % 50 == 0 or i == len(missing):
                elapsed = int(time.time() - start)
                rate = i / elapsed if elapsed > 0 else 0
                eta = int((len(missing) - i) / rate) if rate > 0 else 0
                print(
                    f"  [{i}/{len(missing)}] "
                    f"ok={success_count} fail={fail_count} "
                    f"({elapsed}s elapsed, ~{eta}s remaining)"
                )

                # Save failed log incrementally
                with open(FAILED_LOG, "w", encoding="utf-8") as f:
                    json.dump(list(set(failed_ids)), f)

    elapsed = int(time.time() - start)
    total_transcripts = len(list(WHISPER_DIR.glob("*.txt")))

    print(f"\nDone in {elapsed // 60}m {elapsed % 60}s")
    print(f"  Transcribed: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total whisper transcripts on disk: {total_transcripts}")


def print_stats():
    videos = load_raw_videos()
    total = len(videos)
    have = len(list(WHISPER_DIR.glob("*.txt")))
    pids = set(v.get("post_id", "") for v in videos if v.get("post_id"))
    have_ids = set(f.stem for f in WHISPER_DIR.glob("*.txt"))
    covered = len(pids & have_ids)
    missing = len(pids - have_ids)

    failed = 0
    if FAILED_LOG.exists():
        with open(FAILED_LOG, "r", encoding="utf-8") as f:
            failed = len(json.load(f))

    print(f"Raw videos:              {total}")
    print(f"Whisper transcripts:     {have}")
    print(f"Coverage:                {covered}/{len(pids)} ({covered/len(pids)*100:.1f}%)")
    print(f"Missing:                 {missing}")
    print(f"Previously failed:       {failed}")
    print(f"Actionable (not failed): {missing - failed}")


def main():
    parser = argparse.ArgumentParser(description="Whisper transcription for TikTok videos")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers (default 5)")
    parser.add_argument("--test", type=int, default=None, help="Test on N videos only")
    parser.add_argument("--stats", action="store_true", help="Print stats and exit")
    parser.add_argument("--retry-failed", action="store_true", help="Clear failed log and retry")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.retry_failed and FAILED_LOG.exists():
        FAILED_LOG.unlink()
        print("Cleared failed log. Will retry all previously failed videos.")

    run_transcription(workers=args.workers, max_videos=args.test)


if __name__ == "__main__":
    main()
