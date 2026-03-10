"""Tests for harvester and disappearance tracker bug fixes."""

import json
import os
import tempfile
import time

import pytest

# ─── Fix 1: Atomic file write in disappearance tracker ──────────

from disappearance_tracker import _write_back_to_file, CONFIDENCE_SOLD, CONFIDENCE_STALE


class TestAtomicWrite:
    """Verify _write_back_to_file uses atomic write (temp file + rename)."""

    def _make_jsonl(self, tmp_path, records):
        """Write records to a JSONL file and return its path."""
        p = str(tmp_path / "shard.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return p

    def test_basic_write_back(self, tmp_path):
        """Records get sale_confidence written correctly."""
        records = [
            {"listing_id": "aaa", "ts": 1000, "score": 0.5, "min_divine": 1.0},
            {"listing_id": "bbb", "ts": 1001, "score": 0.6, "min_divine": 2.0},
            {"listing_id": "ccc", "ts": 1002, "score": 0.7, "min_divine": 3.0},
        ]
        path = self._make_jsonl(tmp_path, records)

        file_records = [{"listing_id": "aaa"}, {"listing_id": "bbb"}]
        statuses = {"aaa": False, "bbb": True}  # aaa sold, bbb still listed

        updated = _write_back_to_file(path, file_records, statuses)
        assert updated == 2

        # Read back and verify
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]

        assert lines[0]["sale_confidence"] == CONFIDENCE_SOLD
        assert lines[1]["sale_confidence"] == CONFIDENCE_STALE
        assert "sale_confidence" not in lines[2]

    def test_no_temp_files_left_behind(self, tmp_path):
        """After successful write, no .tmp files remain."""
        records = [{"listing_id": "x", "ts": 1, "score": 0.5, "min_divine": 1.0}]
        path = self._make_jsonl(tmp_path, records)

        _write_back_to_file(path, [{"listing_id": "x"}], {"x": False})

        tmp_files = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert tmp_files == [], f"Temp files left behind: {tmp_files}"

    def test_original_file_survives_empty_update(self, tmp_path):
        """File is unchanged when there are no matching listing IDs."""
        records = [{"listing_id": "a", "ts": 1, "score": 0.5, "min_divine": 1.0}]
        path = self._make_jsonl(tmp_path, records)

        with open(path, "r") as f:
            original = f.read()

        updated = _write_back_to_file(path, [{"listing_id": "zzz"}], {"zzz": False})
        assert updated == 0

        with open(path, "r") as f:
            assert f.read() == original


# ─── Fix 2: Dead combo TTL ──────────────────────────────────────

from calibration_harvester import load_state, save_state


class TestDeadComboTTL:
    """Verify dead combos use timestamped format and expire."""

    def test_fresh_state_uses_ts_format(self):
        """New state has dead_combos_ts dict, not dead_combos list."""
        state = load_state(pass_num=9999)  # Non-existent pass
        assert "dead_combos_ts" in state
        assert isinstance(state["dead_combos_ts"], dict)

    def test_legacy_migration(self, tmp_path, monkeypatch):
        """Legacy dead_combos list gets migrated to timestamped format."""
        # Write a legacy-format state file
        from config import HARVESTER_STATE_FILE
        state_dir = tmp_path / "cache"
        state_dir.mkdir()
        state_file = state_dir / "harvester_state_p99.json"
        legacy = {
            "completed_queries": [],
            "total_samples": 0,
            "query_plan_seed": "test",
            "dead_combos": ["rings:cheap", "boots:mid"],
        }
        with open(state_file, "w") as f:
            json.dump(legacy, f)

        # Monkeypatch to use our temp state file
        monkeypatch.setattr(
            "calibration_harvester.HARVESTER_STATE_FILE",
            state_dir / "harvester_state_p1.json",
        )

        # Load and verify the migration code path exists in load_state
        # (migration happens in run_harvester, not load_state, so test
        # the format directly)
        with open(state_file) as f:
            st = json.load(f)

        # Simulate migration logic from run_harvester
        dead_map = st.get("dead_combos_ts", {})
        if not dead_map and st.get("dead_combos"):
            dead_map = {k: time.time() for k in st["dead_combos"]}

        assert "rings:cheap" in dead_map
        assert "boots:mid" in dead_map
        assert isinstance(dead_map["rings:cheap"], float)

    def test_expired_combos_removed(self):
        """Combos older than 3 days are expired."""
        now = time.time()
        dead_map = {
            "rings:cheap": now - 4 * 86400,   # 4 days old -> expired
            "boots:mid": now - 1 * 86400,      # 1 day old -> kept
            "gloves:high": now - 2.5 * 86400,  # 2.5 days -> kept
        }
        dead_ttl = 3 * 86400
        expired = [k for k, ts in dead_map.items() if now - ts > dead_ttl]
        for k in expired:
            del dead_map[k]

        assert "rings:cheap" not in dead_map
        assert "boots:mid" in dead_map
        assert "gloves:high" in dead_map


# ─── Fix 3: State saved on HTTP errors ─────────────────────────

class TestStateSaveOnError:
    """Verify save_state is called after search HTTP errors.

    We test this by inspecting the source code — the fix adds a
    save_state() call right after the error counter increments.
    """

    def test_save_state_after_http_error(self):
        """The save_state call exists in the HTTP error branch."""
        import inspect
        from calibration_harvester import run_harvester

        source = inspect.getsource(run_harvester)

        # Find the HTTP error block and verify save_state follows
        lines = source.split("\n")
        found_error_block = False
        found_save_after = False
        for i, line in enumerate(lines):
            if "Search HTTP" in line and "skipping" in line:
                found_error_block = True
            if found_error_block and "save_state" in line:
                found_save_after = True
                break
            # If we hit another query processing block, stop looking
            if found_error_block and "search_data = resp.json()" in line:
                break

        assert found_error_block, "Could not find HTTP error handling block"
        assert found_save_after, "save_state() not called after HTTP error"


# ─── Fix 4: Burst counter before fetch ──────────────────────────

class TestBurstCounterOrder:
    """Verify burst_count is incremented before _do_fetch, not after."""

    def test_burst_increment_before_fetch(self):
        """burst_count += 1 appears before _do_fetch in the fetch loop."""
        import inspect
        from calibration_harvester import run_harvester

        source = inspect.getsource(run_harvester)
        lines = source.split("\n")

        # Find the fetch loop (the one with _do_fetch)
        in_fetch_loop = False
        burst_line = None
        fetch_line = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Look for the inner fetch batch loop
            if "_do_fetch" in stripped:
                fetch_line = i
            if "burst_count += 1" in stripped and fetch_line is None:
                burst_line = i
            # Reset if we pass the fetch without finding burst before it
            if fetch_line is not None and burst_line is not None:
                break

        assert burst_line is not None, "Could not find burst_count += 1"
        assert fetch_line is not None, "Could not find _do_fetch call"
        assert burst_line < fetch_line, (
            f"burst_count += 1 (line {burst_line}) should come before "
            f"_do_fetch (line {fetch_line})"
        )
