"""
Batch extract reference audio for all celebrities from the manifest.

Reads celebrities/reference_manifest.yaml, downloads + extracts + converts
each clip, then auto-transcribes with mlx_whisper and updates the YAML configs.

Usage:
  cd /Users/gregorygorin/Projects/voice/mlx-tts
  source .venv/bin/activate
  python ../MOSS-TTS/tools/batch_extract_references.py                          # all
  python ../MOSS-TTS/tools/batch_extract_references.py --celebrity morgan_freeman  # one
  python ../MOSS-TTS/tools/batch_extract_references.py --skip-transcribe        # no whisper
  python ../MOSS-TTS/tools/batch_extract_references.py --list                   # show status
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = SCRIPT_DIR / "celebrities" / "reference_manifest.yaml"
AUDIO_DIR = SCRIPT_DIR / "celebrities" / "audio"
CONFIGS_DIR = SCRIPT_DIR / "celebrities" / "configs"
SAMPLE_RATE = 24000


def check_dependencies():
    missing = []
    for cmd in ["yt-dlp", "ffmpeg"]:
        if not shutil.which(cmd):
            missing.append(cmd)
    if missing:
        print(f"ERROR: Missing: {', '.join(missing)}")
        print("Install with: brew install yt-dlp ffmpeg")
        sys.exit(1)


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def download_and_extract(url, start, end, output_path):
    """Download from YouTube, extract time range, convert to 24kHz mono WAV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "download"
        # Download
        cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
               "-o", str(raw_path), url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"yt-dlp failed: {result.stderr[:200]}"

        # Find downloaded file
        candidates = list(Path(tmpdir).glob("download.*"))
        if not candidates:
            return False, "No downloaded file found"
        downloaded = candidates[0]

        # Extract + convert
        duration = end - start
        cmd = ["ffmpeg", "-y", "-i", str(downloaded),
               "-ss", str(start), "-t", str(duration),
               "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-sample_fmt", "s16", "-acodec", "pcm_s16le",
               str(output_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"ffmpeg failed: {result.stderr[:200]}"

    return True, None


def transcribe(audio_path):
    """Transcribe audio with mlx_whisper."""
    try:
        import mlx_whisper
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
        )
        return result["text"].strip()
    except ImportError:
        return None
    except Exception as e:
        return None


def update_config(slug, ref_text):
    """Update the celebrity's YAML config with the transcribed ref_text."""
    config_path = CONFIGS_DIR / f"{slug}.yaml"
    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["ref_text"] = ref_text

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  width=80, sort_keys=False)

    return True


def get_audio_duration(path):
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return float(result.stdout.strip())
    return 0


def show_status(manifest):
    print(f"\n{'='*70}")
    print(f"  Reference Audio Status")
    print(f"{'='*70}")

    ready = 0
    needs_source = 0
    needs_extract = 0

    for slug, data in manifest["celebrities"].items():
        ref_path = AUDIO_DIR / f"reference_{slug}.wav"
        sources = data.get("sources", [])

        if ref_path.exists():
            dur = get_audio_duration(ref_path)
            config_path = CONFIGS_DIR / f"{slug}.yaml"
            has_text = False
            if config_path.exists():
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                has_text = "PLACEHOLDER" not in str(cfg.get("ref_text", "PLACEHOLDER"))

            status = "READY" if has_text else "needs transcription"
            print(f"  {slug:<28} {dur:5.1f}s  [{status}]")
            ready += 1
        elif sources:
            print(f"  {slug:<28}        [has source, needs extract]")
            needs_extract += 1
        else:
            print(f"  {slug:<28}        [needs source URL]")
            needs_source += 1

    print(f"\n  Ready: {ready}  |  Needs extract: {needs_extract}  |  Needs source: {needs_source}")
    print()


def process_celebrity(slug, data, skip_transcribe=False):
    """Extract, transcribe, and update config for one celebrity."""
    sources = data.get("sources", [])
    if not sources:
        print(f"  {slug}: no sources in manifest, skipping")
        return False

    ref_path = AUDIO_DIR / f"reference_{slug}.wav"
    if ref_path.exists():
        print(f"  {slug}: reference already exists, skipping")
        return True

    # Use first source
    source = sources[0]
    url = source["url"]
    start = source["start"]
    end = source["end"]
    title = source.get("title", "unknown")

    print(f"\n  {slug}: extracting from \"{title}\"")
    print(f"    URL: {url}")
    print(f"    Range: {start}s - {end}s ({end-start:.1f}s)")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ok, err = download_and_extract(url, start, end, ref_path)

    if not ok:
        print(f"    ERROR: {err}")
        return False

    dur = get_audio_duration(ref_path)
    print(f"    Extracted: {dur:.1f}s, 24kHz mono WAV")

    if not skip_transcribe:
        print(f"    Transcribing...")
        text = transcribe(ref_path)
        if text:
            print(f"    Transcript: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
            update_config(slug, text)
            print(f"    Config updated: celebrities/configs/{slug}.yaml")
        else:
            print(f"    WARNING: Transcription failed — update ref_text manually")

    return True


def main():
    parser = argparse.ArgumentParser(description="Batch extract celebrity reference audio")
    parser.add_argument("--celebrity", help="Process only this celebrity")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip whisper transcription")
    parser.add_argument("--list", action="store_true", help="Show extraction status")
    args = parser.parse_args()

    check_dependencies()
    manifest = load_manifest()

    if args.list:
        show_status(manifest)
        return

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if args.celebrity:
        if args.celebrity not in manifest["celebrities"]:
            print(f"ERROR: {args.celebrity} not in manifest")
            sys.exit(1)
        process_celebrity(args.celebrity, manifest["celebrities"][args.celebrity],
                          args.skip_transcribe)
    else:
        results = {"ok": 0, "skip": 0, "fail": 0}
        for slug, data in manifest["celebrities"].items():
            sources = data.get("sources", [])
            ref_path = AUDIO_DIR / f"reference_{slug}.wav"

            if ref_path.exists():
                results["skip"] += 1
                continue
            if not sources:
                results["skip"] += 1
                continue

            ok = process_celebrity(slug, data, args.skip_transcribe)
            if ok:
                results["ok"] += 1
            else:
                results["fail"] += 1

        print(f"\n{'='*70}")
        print(f"  Extracted: {results['ok']}  |  Skipped: {results['skip']}  |  Failed: {results['fail']}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
