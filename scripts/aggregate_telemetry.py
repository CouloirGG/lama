"""
Aggregate telemetry uploads from Discord into a calibration shard.

Reads .json.gz attachments from the #telemetry Discord channel (via bot token),
expands them into raw JSONL records, then feeds them through shard_generator.py
for quality filtering, outlier removal, dedup, and shard output.

Optionally purges processed messages from the channel after aggregation.

Requirements:
    DISCORD_BOT_TOKEN env var (bot needs Read Message History + Manage Messages)
    DISCORD_TELEMETRY_WEBHOOK_URL env var (used to resolve channel ID)

Usage:
    python scripts/aggregate_telemetry.py
    python scripts/aggregate_telemetry.py --purge          # delete processed messages
    python scripts/aggregate_telemetry.py --local DIR      # use downloaded .json.gz files
    python scripts/aggregate_telemetry.py --validate       # validate output shard
"""

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add src/ to path for shard_generator imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import requests

DISCORD_API = "https://discord.com/api/v10"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "telemetry"
MERGED_JSONL = OUTPUT_DIR / "telemetry_merged.jsonl"


def get_channel_id(webhook_url: str) -> str:
    """Resolve channel ID from a webhook URL."""
    resp = requests.get(webhook_url, timeout=10)
    resp.raise_for_status()
    return resp.json()["channel_id"]


def fetch_messages(channel_id: str, bot_token: str, limit: int = 500) -> list:
    """Fetch up to `limit` messages from a Discord channel (newest first)."""
    headers = {"Authorization": f"Bot {bot_token}"}
    messages = []
    before = None

    while len(messages) < limit:
        params = {"limit": min(100, limit - len(messages))}
        if before:
            params["before"] = before
        resp = requests.get(
            f"{DISCORD_API}/channels/{channel_id}/messages",
            headers=headers, params=params, timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        messages.extend(batch)
        before = batch[-1]["id"]
        if len(batch) < 100:
            break
        time.sleep(0.5)  # respect rate limits

    return messages


def download_attachments(messages: list, dest_dir: Path) -> list[Path]:
    """Download .json.gz attachments from Discord messages. Returns file paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for msg in messages:
        for att in msg.get("attachments", []):
            name = att["filename"]
            if not name.endswith(".json.gz"):
                continue
            # Include message ID in filename to avoid collisions
            local_name = f"{msg['id']}_{name}"
            local_path = dest_dir / local_name
            if local_path.exists():
                files.append(local_path)
                continue
            resp = requests.get(att["url"], timeout=30)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            files.append(local_path)
    return files


def expand_telemetry_files(gz_files: list[Path]) -> list[dict]:
    """Expand telemetry .json.gz files into raw calibration-style records."""
    records = []
    for gz_path in gz_files:
        try:
            with gzip.open(gz_path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"  Warning: could not read {gz_path.name}: {e}")
            continue

        version = payload.get("v", "?")
        league = payload.get("league", "")
        samples = payload.get("samples", [])

        for s in samples:
            # Convert compact telemetry format back to calibration JSONL format
            records.append({
                "ts": s.get("ts", 0),
                "league": league,
                "grade": s.get("g", ""),
                "score": s.get("s", 0),
                "item_class": s.get("c", ""),
                "min_divine": s.get("p", 0),
                "max_divine": s.get("p", 0),
                "results": s.get("r", 0),
                "estimate": False,
                "dps_factor": s.get("d", 1.0),
                "defense_factor": s.get("f", 1.0),
                "somv_factor": s.get("v", 1.0),
                "source": f"telemetry_{version}",
            })

        print(f"  {gz_path.name}: {len(samples)} samples (v{version}, {league})")

    return records


def purge_messages(channel_id: str, bot_token: str, message_ids: list[str]):
    """Delete processed messages from Discord channel."""
    headers = {"Authorization": f"Bot {bot_token}"}
    deleted = 0

    # Bulk delete (up to 100 at a time, messages < 14 days old)
    recent = [mid for mid in message_ids]
    for i in range(0, len(recent), 100):
        batch = recent[i:i + 100]
        if len(batch) == 1:
            resp = requests.delete(
                f"{DISCORD_API}/channels/{channel_id}/messages/{batch[0]}",
                headers=headers, timeout=10,
            )
            if resp.status_code in (200, 204):
                deleted += 1
            time.sleep(0.5)
        else:
            resp = requests.post(
                f"{DISCORD_API}/channels/{channel_id}/messages/bulk-delete",
                headers=headers, json={"messages": batch}, timeout=15,
            )
            if resp.status_code in (200, 204):
                deleted += len(batch)
            elif resp.status_code == 400:
                # Some messages too old for bulk delete — fall back to individual
                for mid in batch:
                    r = requests.delete(
                        f"{DISCORD_API}/channels/{channel_id}/messages/{mid}",
                        headers=headers, timeout=10,
                    )
                    if r.status_code in (200, 204):
                        deleted += 1
                    time.sleep(0.5)
            time.sleep(1)

    print(f"  Purged {deleted}/{len(message_ids)} messages")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate telemetry from Discord into a calibration shard")
    parser.add_argument("--local", metavar="DIR",
                        help="Use local directory of .json.gz files instead of Discord")
    parser.add_argument("--purge", action="store_true",
                        help="Delete processed messages from Discord after aggregation")
    parser.add_argument("--validate", action="store_true",
                        help="Run hold-out validation on the output shard")
    parser.add_argument("--output", "-o",
                        help="Output shard path (default: data/telemetry/telemetry_shard.json.gz)")
    parser.add_argument("--league", default="",
                        help="League name (auto-detected if omitted)")
    parser.add_argument("--limit", type=int, default=500,
                        help="Max Discord messages to fetch (default: 500)")

    args = parser.parse_args()

    output_path = args.output or str(OUTPUT_DIR / "telemetry_shard.json.gz")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Collect .json.gz files ──
    message_ids = []

    if args.local:
        local_dir = Path(args.local)
        gz_files = sorted(local_dir.glob("*.json.gz"))
        print(f"Local mode: found {len(gz_files)} files in {local_dir}")
    else:
        webhook_url = os.environ.get("DISCORD_TELEMETRY_WEBHOOK_URL", "")
        bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
        if not webhook_url:
            print("ERROR: DISCORD_TELEMETRY_WEBHOOK_URL not set")
            sys.exit(1)
        if not bot_token:
            print("ERROR: DISCORD_BOT_TOKEN not set (needed to read channel)")
            sys.exit(1)

        print("Resolving channel ID from webhook...")
        channel_id = get_channel_id(webhook_url)
        print(f"  Channel: {channel_id}")

        print(f"Fetching messages (limit={args.limit})...")
        messages = fetch_messages(channel_id, bot_token, limit=args.limit)
        # Filter to messages with .json.gz attachments
        with_gz = [m for m in messages
                    if any(a["filename"].endswith(".json.gz")
                           for a in m.get("attachments", []))]
        print(f"  Found {len(with_gz)} messages with telemetry attachments")

        if not with_gz:
            print("Nothing to aggregate.")
            return

        message_ids = [m["id"] for m in with_gz]

        print("Downloading attachments...")
        dl_dir = OUTPUT_DIR / "downloads"
        gz_files = download_attachments(with_gz, dl_dir)

    if not gz_files:
        print("No .json.gz files found.")
        return

    # ── Expand into JSONL records ──
    print(f"\nExpanding {len(gz_files)} telemetry files...")
    records = expand_telemetry_files(gz_files)
    print(f"  Total records: {len(records)}")

    if not records:
        print("No records to process.")
        return

    # ── Write merged JSONL (intermediate) ──
    with open(MERGED_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Written: {MERGED_JSONL}")

    # ── Run shard_generator pipeline ──
    from shard_generator import generate_shard, load_raw_records, validate_shard

    print("\n" + "=" * 50)
    print("  Shard Generation Pipeline")
    print("=" * 50)
    shard = generate_shard(records, args.league, output_path)

    # ── Validate if requested ──
    if args.validate:
        print("\n" + "=" * 50)
        validate_shard(output_path)

    # ── Purge processed messages ──
    if args.purge and message_ids:
        print(f"\nPurging {len(message_ids)} processed messages...")
        purge_messages(channel_id, bot_token, message_ids)

    print(f"\nDone. Shard: {output_path}")


if __name__ == "__main__":
    main()
