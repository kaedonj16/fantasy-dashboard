"""Unit tests for graphs scatter label collision helper."""
from __future__ import annotations

from utils.scatter_labels import scatter_label_placements, scatter_label_placements_from_rows


def test_sparse_points_keep_default_top_center_labels():
    # Far apart in normalized space — every label stays visible at top center.
    placed = scatter_label_placements(
        [0.0, 100.0],
        [0.0, 100.0],
        ["Alpha", "Bravo"],
    )
    assert placed == [("Alpha", "top center"), ("Bravo", "top center")]


def test_dense_cluster_offsets_or_hides_overlapping_labels():
    # Tour-mock style: several teams piled near the same PF/PA.
    xs = [50.0, 50.5, 51.0, 50.2, 50.8, 49.8]
    ys = [100.0, 100.2, 99.8, 100.1, 99.9, 100.3]
    labels = [f"Team{i}" for i in range(len(xs))]
    placed = scatter_label_placements(xs, ys, labels)

    assert len(placed) == len(labels)
    # First claim wins; at least one later label is offset or hidden.
    assert placed[0][0] == "Team0"
    assert placed[0][1] == "top center"
    texts = [t for t, _ in placed]
    positions = [p for _, p in placed]
    assert sum(1 for t in texts if t) < len(labels) or len(set(positions)) > 1
    # Hidden labels use empty text (marker hover still carries the name).
    assert all(isinstance(t, str) and isinstance(p, str) for t, p in placed)


def test_priority_order_keeps_earlier_labels():
    rows = [
        {"x": 10, "y": 10, "label": "Viewer"},
        {"x": 10.1, "y": 10.05, "label": "RivalA"},
        {"x": 10.05, "y": 10.1, "label": "RivalB"},
        {"x": 10.08, "y": 10.02, "label": "RivalC"},
        {"x": 9.95, "y": 10.08, "label": "RivalD"},
    ]
    placed = scatter_label_placements_from_rows(rows)
    assert placed[0] == ("Viewer", "top center")
    # With hide_when_crowded, some rivals drop rather than smear the plot.
    visible = [t for t, _ in placed if t]
    assert "Viewer" in visible
    assert len(visible) < len(rows)


def test_mismatched_lengths_raise():
    try:
        scatter_label_placements([1, 2], [1], ["a", "b"])
    except ValueError as e:
        assert "same length" in str(e)
    else:
        raise AssertionError("expected ValueError")
