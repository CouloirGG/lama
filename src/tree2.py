"""GGG official PoE2 passive-tree export — web-ready data + sprite atlases.

Downloads grindinggear/poe2-skilltree-export (current season branch): the
`data.json` (nodes with precomputed x/y + edges) and the WebP sprite atlases
(node icons, frames, connectors, group backgrounds) on first use, caches them
under resources/data/tree2/, and serves them to the canvas renderer. Also
exposes a notable-name -> node-id index so the backend can flag swap nodes.

This replaces the PoB `.dds.zst` tree, which is stale (0.4) and not
web-renderable. See docs / memory: reference_poe2_tree_render.
"""
import json
import logging
import threading
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

TREE2_BRANCH = "0.5.0"
TREE2_BASE = f"https://raw.githubusercontent.com/grindinggear/poe2-skilltree-export/{TREE2_BRANCH}/"
TREE2_DIR = Path(__file__).parent.parent / "resources" / "data" / "tree2"

# Files served to the browser canvas (relative paths preserved under the cache).
TREE2_FILES = [
    "data.json",
    "assets/skills.webp", "assets/skills.json",
    "assets/skills-disabled.webp", "assets/skills-disabled.json",
    "assets/frame.webp", "assets/frame.json",
    "assets/line.webp", "assets/line.json",
    "assets/group-background.webp", "assets/group-background.json",
]

_lock = threading.Lock()
_data = None
_name_index = None


def ensure_assets(force=False):
    """Download any missing tree-export files into the cache dir."""
    TREE2_DIR.mkdir(parents=True, exist_ok=True)
    for rel in TREE2_FILES:
        dest = TREE2_DIR / rel
        if dest.exists() and dest.stat().st_size > 0 and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            req = urllib.request.Request(TREE2_BASE + rel, headers={"User-Agent": "LAMA"})
            with urllib.request.urlopen(req, timeout=90) as r:
                blob = r.read()
            dest.write_bytes(blob)
            logger.info(f"[tree2] cached {rel} ({len(blob)} bytes)")
        except Exception as e:
            logger.warning(f"[tree2] failed to fetch {rel}: {e}")


def assets_ready():
    return all((TREE2_DIR / rel).exists() for rel in TREE2_FILES)


def file_path(rel):
    """Resolve a cached tree2 file path, or None if missing / outside the dir."""
    base = TREE2_DIR.resolve()
    p = (TREE2_DIR / rel.lstrip("/")).resolve()
    try:
        p.relative_to(base)
    except ValueError:
        return None
    return p if p.exists() else None


def _load():
    global _data, _name_index
    if _data is not None:
        return
    with _lock:
        if _data is not None:
            return
        ensure_assets()
        dpath = TREE2_DIR / "data.json"
        if not dpath.exists():
            return
        try:
            _data = json.loads(dpath.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[tree2] data.json load failed: {e}")
            return
        idx = {}
        for nid, n in (_data.get("nodes") or {}).items():
            nm = (n.get("name") or "").strip().lower()
            if nm:
                idx.setdefault(nm, []).append(str(nid))
        _name_index = idx


def names_to_ids(names):
    """Map notable names to GGG node ids (deduped, order-preserving)."""
    _load()
    if not _name_index:
        return []
    seen, out = set(), []
    for nm in names or []:
        for nid in _name_index.get((nm or "").strip().lower(), []):
            if nid not in seen:
                seen.add(nid)
                out.append(nid)
    return out
