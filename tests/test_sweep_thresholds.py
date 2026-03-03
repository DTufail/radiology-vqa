"""Tests for scripts/sweep_thresholds.py — CPU-only, no GPU required.

All tests use synthetic (confidence, correct, answer_type) data so they run
quickly in CI without any model weights or dataset files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to sys.path so we can import sweep_thresholds directly.
_SCRIPTS_DIR = str(Path(__file__).parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import sweep_thresholds as st


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_layout_a(
    n_closed: int = 30,
    n_open: int = 20,
    closed_conf: float = 0.80,
    open_conf: float = 0.55,
    closed_acc: float = 0.70,
    open_acc: float = 0.45,
) -> list[dict]:
    """Build a synthetic Layout A prediction list."""
    records = []
    for i in range(n_closed):
        records.append(
            {
                "confidence": closed_conf,
                "correct": i < int(n_closed * closed_acc),
                "answer_type": "closed",
            }
        )
    for i in range(n_open):
        records.append(
            {
                "confidence": open_conf,
                "correct": i < int(n_open * open_acc),
                "answer_type": "open",
            }
        )
    return records


def _row(
    high_t: float = 0.65,
    low_t: float = 0.35,
    abstain_rate: float = 0.15,
    accuracy_when_answered: float = 0.80,
    overall_accuracy: float = 0.68,
    n_samples: int = 50,
) -> dict:
    return {
        "high_t": high_t,
        "low_t": low_t,
        "abstain_rate": abstain_rate,
        "accuracy_when_answered": accuracy_when_answered,
        "overall_accuracy": overall_accuracy,
        "n_samples": n_samples,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TestLoadResults
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadResults:
    """Tests for _load_results()."""

    def test_layout_a_returns_three_lists(self, tmp_path: Path):
        """Layout A returns (confs, corrects, answer_types) of equal length."""
        records = _make_layout_a(n_closed=10, n_open=5)
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))
        confs, corrects, types = st._load_results(p)
        assert len(confs) == 15
        assert len(corrects) == 15
        assert len(types) == 15

    def test_layout_a_conf_values_are_floats(self, tmp_path: Path):
        """Confidence values are cast to float."""
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"confidence": "0.75", "correct": True, "answer_type": "closed"}]))
        confs, _, _ = st._load_results(p)
        assert isinstance(confs[0], float)
        assert confs[0] == pytest.approx(0.75)

    def test_layout_a_correct_values_are_bools(self, tmp_path: Path):
        """Correctness values are cast to bool."""
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"confidence": 0.80, "correct": 1, "answer_type": "open"}]))
        _, corrects, _ = st._load_results(p)
        assert isinstance(corrects[0], bool)
        assert corrects[0] is True

    def test_layout_a_null_answer_type_defaults_open(self, tmp_path: Path):
        """answer_type=null defaults to 'open'."""
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"confidence": 0.70, "correct": True, "answer_type": None}]))
        _, _, types = st._load_results(p)
        assert types[0] == "open"

    def test_layout_a_missing_answer_type_defaults_open(self, tmp_path: Path):
        """Missing answer_type key defaults to 'open'."""
        p = tmp_path / "r.json"
        p.write_text(json.dumps([{"confidence": 0.70, "correct": False}]))
        _, _, types = st._load_results(p)
        assert types[0] == "open"

    def test_layout_b_returns_sentinel(self, tmp_path: Path):
        """Layout B (dict with 'combined') returns ([], [], []) sentinel."""
        data = {"combined": [_row()], "closed": [], "open": []}
        p = tmp_path / "sweep.json"
        p.write_text(json.dumps(data))
        confs, corrects, types = st._load_results(p)
        assert confs == []
        assert corrects == []
        assert types == []

    def test_file_not_found_raises(self, tmp_path: Path):
        """FileNotFoundError is raised for a missing file."""
        with pytest.raises(FileNotFoundError):
            st._load_results(tmp_path / "nonexistent.json")

    def test_missing_required_keys_raises(self, tmp_path: Path):
        """ValueError is raised when required keys are absent."""
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"answer_type": "closed"}]))
        with pytest.raises(ValueError, match="missing required keys"):
            st._load_results(p)


# ══════════════════════════════════════════════════════════════════════════════
# TestRunSweepGrid
# ══════════════════════════════════════════════════════════════════════════════


class TestRunSweepGrid:
    """Tests for _run_sweep_grid()."""

    def test_returns_list_of_dicts(self):
        """Returns a non-empty list of dicts for non-empty input."""
        confs    = [0.80] * 10
        corrects = [True] * 7 + [False] * 3
        rows = st._run_sweep_grid(confs, corrects)
        assert isinstance(rows, list)
        assert len(rows) > 0
        assert all(isinstance(r, dict) for r in rows)

    def test_row_has_required_keys(self):
        """Every row has exactly the required keys."""
        required = {
            "high_t", "low_t", "abstain_rate",
            "accuracy_when_answered", "overall_accuracy", "n_samples",
        }
        rows = st._run_sweep_grid([0.70] * 5, [True] * 5)
        for r in rows:
            assert set(r.keys()) == required

    def test_empty_input_returns_empty(self):
        """Empty input returns an empty list."""
        assert st._run_sweep_grid([], []) == []

    def test_n_samples_equals_input_length(self):
        """n_samples in every row equals len(confs)."""
        n = 17
        rows = st._run_sweep_grid([0.60] * n, [True] * n)
        for r in rows:
            assert r["n_samples"] == n

    def test_low_t_strictly_less_than_high_t(self):
        """No row has low_t >= high_t."""
        rows = st._run_sweep_grid([0.50] * 5, [True] * 5)
        for r in rows:
            assert r["low_t"] < r["high_t"]

    def test_all_abstain_when_conf_below_grid_min(self):
        """All samples abstain when every conf is below the smallest low_t (0.10)."""
        confs    = [0.05] * 10
        corrects = [True] * 10
        rows = st._run_sweep_grid(confs, corrects)
        for r in rows:
            assert r["abstain_rate"] == pytest.approx(1.0), (
                f"Expected abstain_rate=1 for low_t={r['low_t']}, conf=0.05"
            )

    def test_accuracy_when_answered_bounded(self):
        """accuracy_when_answered is always in [0, 1]."""
        import random
        random.seed(42)
        confs    = [random.uniform(0.3, 0.9) for _ in range(40)]
        corrects = [random.choice([True, False]) for _ in range(40)]
        rows = st._run_sweep_grid(confs, corrects)
        for r in rows:
            assert 0.0 <= r["accuracy_when_answered"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TestFindBestRow
# ══════════════════════════════════════════════════════════════════════════════


class TestFindBestRow:
    """Tests for _find_best_row()."""

    def test_empty_rows_returns_empty_dict(self):
        """Empty rows list returns an empty dict."""
        assert st._find_best_row([], 0.10, 0.25) == {}

    def test_returns_row_in_target_range(self):
        """Prefers rows inside the [target_min, target_max] window."""
        rows = [
            _row(low_t=0.35, abstain_rate=0.15, accuracy_when_answered=0.80),
            _row(low_t=0.10, abstain_rate=0.02, accuracy_when_answered=0.75),
        ]
        best = st._find_best_row(rows, 0.10, 0.25)
        # Only the first row has abstain_rate in [0.10, 0.25]
        assert best["low_t"] == pytest.approx(0.35)

    def test_fallback_when_no_target_range(self):
        """Falls back to all rows when none are in the target window."""
        rows = [
            _row(low_t=0.10, abstain_rate=0.02, accuracy_when_answered=0.80),
            _row(low_t=0.50, abstain_rate=0.60, accuracy_when_answered=0.90),
        ]
        # Neither has abstain_rate in [0.10, 0.25] — should pick highest accuracy.
        best = st._find_best_row(rows, 0.10, 0.25)
        assert best["accuracy_when_answered"] == pytest.approx(0.90)

    def test_returns_dict_with_required_keys(self):
        """The returned dict has the standard row keys."""
        rows = [_row()]
        best = st._find_best_row(rows, 0.10, 0.25)
        required = {
            "high_t", "low_t", "abstain_rate",
            "accuracy_when_answered", "overall_accuracy", "n_samples",
        }
        assert required.issubset(set(best.keys()))


# ══════════════════════════════════════════════════════════════════════════════
# TestWritePhase7aYaml
# ══════════════════════════════════════════════════════════════════════════════


class TestWritePhase7aYaml:
    """Tests for _write_phase7a_yaml()."""

    def test_creates_file(self, tmp_path: Path):
        """Creates the YAML file at the given path."""
        out = tmp_path / "phase7a.yaml"
        st._write_phase7a_yaml(out, _row(0.65, 0.20), _row(0.60, 0.35), _row(0.65, 0.35), "isotonic", "data/iso.json")
        assert out.exists()

    def test_yaml_contains_calibration_method(self, tmp_path: Path):
        """calibration_method: "isotonic" appears in the file content."""
        out = tmp_path / "phase7a.yaml"
        st._write_phase7a_yaml(out, _row(), _row(), _row(), "isotonic", "data/iso.json")
        assert 'calibration_method: "isotonic"' in out.read_text()

    def test_yaml_contains_closed_low_confidence(self, tmp_path: Path):
        """supervisor_closed_low_confidence uses the closed row's low_t."""
        out = tmp_path / "phase7a.yaml"
        closed = _row(high_t=0.65, low_t=0.20)
        open_  = _row(high_t=0.60, low_t=0.35)
        st._write_phase7a_yaml(out, closed, open_, _row(), "isotonic", "")
        assert "supervisor_closed_low_confidence: 0.20" in out.read_text()

    def test_yaml_contains_open_low_confidence(self, tmp_path: Path):
        """supervisor_open_low_confidence uses the open row's low_t."""
        out = tmp_path / "phase7a.yaml"
        closed = _row(high_t=0.65, low_t=0.20)
        open_  = _row(high_t=0.60, low_t=0.35)
        st._write_phase7a_yaml(out, closed, open_, _row(), "isotonic", "")
        assert "supervisor_open_low_confidence: 0.35" in out.read_text()

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Creates non-existent parent directories."""
        out = tmp_path / "nested" / "deep" / "phase7a.yaml"
        st._write_phase7a_yaml(out, _row(), _row(), _row(), "isotonic", "")
        assert out.exists()


# ══════════════════════════════════════════════════════════════════════════════
# TestSweepPipeline
# ══════════════════════════════════════════════════════════════════════════════


class TestSweepPipeline:
    """End-to-end pipeline tests using synthetic data."""

    def test_closed_higher_acc_than_open(self, tmp_path: Path):
        """With synthetic data, closed subset should have higher accuracy than open."""
        records = _make_layout_a(
            n_closed=50,
            n_open=50,
            closed_conf=0.80,
            open_conf=0.55,
            closed_acc=0.90,
            open_acc=0.40,
        )
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))
        confs, corrects, types = st._load_results(p)

        closed_confs    = [c  for c, t in zip(confs,    types) if t == "closed"]
        closed_corrects = [ok for ok, t in zip(corrects, types) if t == "closed"]
        open_confs      = [c  for c, t in zip(confs,    types) if t == "open"]
        open_corrects   = [ok for ok, t in zip(corrects, types) if t == "open"]

        rows_closed = st._run_sweep_grid(closed_confs, closed_corrects)
        rows_open   = st._run_sweep_grid(open_confs,   open_corrects)

        best_closed = st._find_best_row(rows_closed, 0.10, 0.25)
        best_open   = st._find_best_row(rows_open,   0.10, 0.25)

        assert (
            best_closed.get("accuracy_when_answered", 0)
            > best_open.get("accuracy_when_answered", 0)
        )

    def test_n_samples_matches_input(self, tmp_path: Path):
        """n_samples in sweep rows equals the length of the input subset."""
        records = _make_layout_a(n_closed=30, n_open=20)
        p = tmp_path / "results.json"
        p.write_text(json.dumps(records))
        confs, corrects, types = st._load_results(p)

        closed_confs    = [c  for c, t in zip(confs,    types) if t == "closed"]
        closed_corrects = [ok for ok, t in zip(corrects, types) if t == "closed"]

        rows = st._run_sweep_grid(closed_confs, closed_corrects)
        for r in rows:
            assert r["n_samples"] == 30
