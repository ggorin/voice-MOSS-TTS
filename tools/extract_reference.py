"""
Extract reference audio clips from YouTube for celebrity voice cloning.

Wraps yt-dlp + ffmpeg to download, extract a time range, and convert
to the required format (mono 24kHz PCM 16-bit WAV).

Usage:
  python tools/extract_reference.py \
    --url "https://youtube.com/watch?v=XXXXX" \
    --start 120.0 --end 135.0 \
    --celebrity morgan_freeman \
    --candidate 1

  # From a local audio/video file:
  python tools/extract_reference.py \
    --file interview.mp4 \
    --start 45.0 --end 60.0 \
    --celebrity morgan_freeman \
    --candidate 2

  # List existing candidates for a celebrity:
  python tools/extract_reference.py --celebrity morgan_freeman --list

  # Promote a candidate to the final reference:
  python tools/extract_reference.py --celebrity morgan_freeman --promote 3
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = SCRIPT_DIR / "celebrities" / "audio"
SAMPLE_RATE = 24000


def check_dependencies():
    """Check that yt-dlp and ffmpeg are available."""
    missing = []
    for cmd in ["yt-dlp", "ffmpeg", "ffprobe"]:
        if not shutil.which(cmd):
            missing.append(cmd)
    if missing:
        print(f"ERROR: Missing required tools: {', '.join(missing)}")
        print("Install with: brew install yt-dlp ffmpeg")
        sys.exit(1)


def download_audio(url, output_path):
    """Download audio-only from YouTube."""
    print(f"  Downloading audio from: {url}")
    cmd = [
        "yt-dlp", "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(output_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  yt-dlp error: {result.stderr}")
        sys.exit(1)
    # yt-dlp may add extension, find the actual file
    candidates = list(output_path.parent.glob(f"{output_path.stem}.*"))
    if candidates:
        return candidates[0]
    return output_path


def extract_and_convert(input_path, output_path, start, end):
    """Extract time range and convert to mono 24kHz 16-bit WAV."""
    duration = end - start
    print(f"  Extracting {start:.1f}s - {end:.1f}s ({duration:.1f}s)")
    print(f"  Converting to mono {SAMPLE_RATE}Hz 16-bit WAV")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-ss", str(start),
        "-t", str(duration),
        "-ac", "1",              # mono
        "-ar", str(SAMPLE_RATE), # 24kHz
        "-sample_fmt", "s16",    # 16-bit
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr}")
        sys.exit(1)


def get_audio_info(path):
    """Get duration and format info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-show_entries", "stream=sample_rate,channels,codec_name",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        import json
        info = json.loads(result.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        stream = info.get("streams", [{}])[0]
        return {
            "duration": duration,
            "sample_rate": stream.get("sample_rate", "?"),
            "channels": stream.get("channels", "?"),
            "codec": stream.get("codec_name", "?"),
        }
    return None


def list_candidates(slug):
    """List existing candidate reference clips for a celebrity."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(AUDIO_DIR.glob(f"reference_{slug}_candidate*.wav"))
    final = AUDIO_DIR / f"reference_{slug}.wav"

    print(f"\n{'='*60}")
    print(f"  Reference audio for: {slug}")
    print(f"{'='*60}")

    if final.exists():
        info = get_audio_info(final)
        dur = f"{info['duration']:.1f}s" if info else "?"
        print(f"  FINAL: {final.name} ({dur})")
    else:
        print(f"  FINAL: not yet selected")

    if candidates:
        for c in candidates:
            info = get_audio_info(c)
            dur = f"{info['duration']:.1f}s" if info else "?"
            print(f"  {c.name} ({dur})")
    else:
        print(f"  No candidates found")
    print()


def promote_candidate(slug, candidate_num):
    """Copy a candidate to the final reference file."""
    src = AUDIO_DIR / f"reference_{slug}_candidate{candidate_num}.wav"
    dst = AUDIO_DIR / f"reference_{slug}.wav"

    if not src.exists():
        print(f"  ERROR: Candidate not found: {src}")
        sys.exit(1)

    shutil.copy2(src, dst)
    info = get_audio_info(dst)
    dur = f"{info['duration']:.1f}s" if info else "?"
    print(f"  Promoted candidate {candidate_num} to final reference ({dur})")
    print(f"  {dst}")
    print(f"\n  Don't forget to add ref_text to the YAML config!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract reference audio for celebrity voice cloning",
    )
    parser.add_argument("--url", help="YouTube URL to download from")
    parser.add_argument("--file", help="Local audio/video file to extract from")
    parser.add_argument("--start", type=float, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds")
    parser.add_argument("--celebrity", required=True, help="Celebrity slug")
    parser.add_argument("--candidate", type=int, default=1, help="Candidate number")
    parser.add_argument("--list", action="store_true", help="List existing candidates")
    parser.add_argument("--promote", type=int, help="Promote candidate N to final reference")
    args = parser.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_candidates(args.celebrity)
        return

    if args.promote:
        promote_candidate(args.celebrity, args.promote)
        return

    if not args.url and not args.file:
        print("ERROR: Provide --url (YouTube) or --file (local file)")
        sys.exit(1)

    if args.start is None or args.end is None:
        print("ERROR: --start and --end are required for extraction")
        sys.exit(1)

    check_dependencies()

    output_path = AUDIO_DIR / f"reference_{args.celebrity}_candidate{args.candidate}.wav"

    print(f"\n{'='*60}")
    print(f"  Extracting reference audio: {args.celebrity}")
    print(f"  Candidate: {args.candidate}")
    print(f"{'='*60}")

    if args.url:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_path = Path(tmpdir) / "download"
            downloaded = download_audio(args.url, raw_path)
            extract_and_convert(downloaded, output_path, args.start, args.end)
    else:
        local_file = Path(args.file)
        if not local_file.exists():
            print(f"  ERROR: File not found: {local_file}")
            sys.exit(1)
        extract_and_convert(local_file, output_path, args.start, args.end)

    info = get_audio_info(output_path)
    if info:
        print(f"\n  Output: {output_path}")
        print(f"  Duration: {info['duration']:.1f}s")
        print(f"  Format: {info['sample_rate']}Hz, {info['channels']}ch, {info['codec']}")
    else:
        print(f"\n  Output: {output_path}")

    print(f"\n  Next steps:")
    print(f"  1. Listen to the clip and verify it's clean single-speaker audio")
    print(f"  2. Transcribe the audio word-for-word")
    print(f"  3. Add transcription as ref_text in celebrities/configs/{args.celebrity}.yaml")
    print(f"  4. To promote to final: python tools/extract_reference.py --celebrity {args.celebrity} --promote {args.candidate}")
    print()


if __name__ == "__main__":
    main()
