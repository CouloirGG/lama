"""
LAMA — Release changelog generator

Generate a structured changelog from commits between main and dev,
optionally post to Discord and/or write GitHub release notes.

Usage:
    python scripts/release.py                  # Interactive mode
    python scripts/release.py --post           # Post to Discord
    python scripts/release.py --write          # Write .github/RELEASE_NOTES.md
    python scripts/release.py --post --write   # Both
    python scripts/release.py --dry-run        # Show what would happen
    python scripts/release.py --include-docs   # Include doc/TODO commits
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import os

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args: str) -> str:
    """Run a git command and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"git error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def get_version() -> str:
    return (ROOT / "resources" / "VERSION").read_text().strip()


def get_commits(base: str = "main", head: str = "dev") -> list[dict]:
    """Return list of {hash, subject} dicts for commits in head but not base."""
    log = git("log", "--oneline", f"{base}..{head}")
    if not log:
        return []
    commits = []
    for line in log.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            commits.append({"hash": parts[0], "subject": parts[1]})
    return commits


# ---------------------------------------------------------------------------
# Categorisation
# ---------------------------------------------------------------------------

# Patterns checked in order; first match wins
CATEGORIES = [
    ("New Features",  re.compile(r"^(Add|Implement)\b", re.IGNORECASE)),
    ("Bug Fixes",     re.compile(r"^(Fix|Revert)\b", re.IGNORECASE)),
    ("Improvements",  re.compile(r"^Update\b(?!.*\b(docs?|TODO)\b)", re.IGNORECASE)),
]

SKIP_PATTERN = re.compile(
    r"^(Update docs|Update TODO|Bump version)\b", re.IGNORECASE,
)


def categorise(
    commits: list[dict], include_docs: bool = False,
) -> dict[str, list[dict]]:
    """Sort commits into categories. Returns {category: [commit, ...]}."""
    result: dict[str, list[dict]] = {}
    for commit in commits:
        subj = commit["subject"]

        # Skip noise commits unless --include-docs
        if not include_docs and SKIP_PATTERN.match(subj):
            continue

        matched = False
        for cat_name, pattern in CATEGORIES:
            if pattern.match(subj):
                result.setdefault(cat_name, []).append(commit)
                matched = True
                break

        if not matched:
            result.setdefault("Other", []).append(commit)

    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_markdown(version: str, categories: dict[str, list[dict]]) -> str:
    """Render a Markdown changelog."""
    lines = [f"## LAMA v{version}", ""]
    for cat_name in ("New Features", "Bug Fixes", "Improvements", "Other"):
        items = categories.get(cat_name)
        if not items:
            continue
        lines.append(f"### {cat_name}")
        for c in items:
            # Strip the verb prefix for cleaner reading
            lines.append(f"- {c['subject']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_discord_embed(
    version: str, categories: dict[str, list[dict]],
) -> dict:
    """Build a Discord webhook payload with a rich embed."""
    fields = []
    for cat_name in ("New Features", "Bug Fixes", "Improvements", "Other"):
        items = categories.get(cat_name)
        if not items:
            continue
        value = "\n".join(f"- {c['subject']}" for c in items)
        # Discord field value max is 1024 chars
        if len(value) > 1024:
            value = value[:1020] + "\n..."
        fields.append({"name": cat_name, "value": value, "inline": False})

    embed = {
        "title": f"LAMA v{version}",
        "color": 0xFFD700,  # Gold
        "fields": fields,
        "footer": {
            "text": "Download at github.com/Couloir/LAMA/releases",
        },
    }
    return {"embeds": [embed]}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def post_to_discord(payload: dict, dry_run: bool = False) -> bool:
    """Post the embed payload to the Discord release webhook."""
    if dry_run:
        print("\n--- Discord payload (dry-run) ---")
        print(json.dumps(payload, indent=2))
        print("--- end ---\n")
        return True

    url = os.environ.get("DISCORD_RELEASE_WEBHOOK_URL", "")
    if not url:
        print("ERROR: DISCORD_RELEASE_WEBHOOK_URL not set in .env")
        return False

    import requests
    resp = requests.post(url, json=payload, timeout=15)
    if resp.status_code in (200, 204):
        return True
    print(f"Discord error {resp.status_code}: {resp.text}", file=sys.stderr)
    return False


def write_release_notes(markdown: str) -> Path:
    """Write markdown to .github/RELEASE_NOTES.md."""
    out = ROOT / ".github" / "RELEASE_NOTES.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a release changelog from main..dev commits",
    )
    parser.add_argument(
        "--post", action="store_true",
        help="Post changelog to Discord",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="Write .github/RELEASE_NOTES.md",
    )
    parser.add_argument(
        "--include-docs", action="store_true",
        help="Include doc/TODO/version-bump commits",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show actions without executing them",
    )
    parser.add_argument(
        "--base", default="main",
        help="Base branch (default: main)",
    )
    parser.add_argument(
        "--head", default="dev",
        help="Head branch (default: dev)",
    )
    args = parser.parse_args()

    version = get_version()
    commits = get_commits(args.base, args.head)

    if not commits:
        print(f"No commits found between {args.base}..{args.head}")
        sys.exit(0)

    print(f"Generating changelog for {args.base}..{args.head} "
          f"({len(commits)} commits)...\n")

    categories = categorise(commits, args.include_docs)
    markdown = format_markdown(version, categories)

    print(f"=== LAMA v{version} ===\n")
    print(markdown)

    # Interactive mode if no flags given
    interactive = not (args.post or args.write)

    # --- Discord ---
    do_post = args.post
    if interactive and not do_post:
        answer = input("Post to Discord? [y/N]: ").strip().lower()
        do_post = answer in ("y", "yes")

    if do_post:
        payload = format_discord_embed(version, categories)
        if post_to_discord(payload, dry_run=args.dry_run):
            print("  Posted to Discord" if not args.dry_run
                  else "  Would post to Discord (dry-run)")
        else:
            print("  Failed to post to Discord", file=sys.stderr)

    # --- Release notes ---
    do_write = args.write
    if interactive and not do_write:
        answer = input("Write release notes for GitHub? [y/N]: ").strip().lower()
        do_write = answer in ("y", "yes")

    if do_write:
        if args.dry_run:
            print("  Would write .github/RELEASE_NOTES.md (dry-run)")
        else:
            path = write_release_notes(markdown)
            print(f"  Wrote {path.relative_to(ROOT)} "
                  "(commit this before tagging)")


if __name__ == "__main__":
    main()
