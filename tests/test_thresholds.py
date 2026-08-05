from aak.analytics.thresholds import (
    DEFAULT_THRESHOLDS_PATH,
    ObservationWindowThresholds,
    ScoringThresholds,
    StagingThresholds,
    StallDetectionThresholds,
    Thresholds,
    load_thresholds,
)


def test_default_thresholds_file_exists():
    assert DEFAULT_THRESHOLDS_PATH.exists()


def test_load_thresholds_returns_typed_object():
    thresholds = load_thresholds()
    assert isinstance(thresholds, Thresholds)
    assert isinstance(thresholds.observation_window, ObservationWindowThresholds)
    assert isinstance(thresholds.staging, StagingThresholds)
    assert isinstance(thresholds.stall_detection, StallDetectionThresholds)
    assert isinstance(thresholds.scoring, ScoringThresholds)


def test_observation_window_values_are_positive():
    thresholds = load_thresholds()
    assert thresholds.observation_window.min_days_general > 0
    assert thresholds.observation_window.min_cohort_size > 0


def test_scoring_weights_cover_all_five_stages():
    thresholds = load_thresholds()
    assert set(thresholds.scoring.stage_weights) == {"notice", "attempt", "navigate", "transform", "embed"}
    assert thresholds.scoring.stage_weights["notice"] == 0
    assert thresholds.scoring.stage_weights["embed"] == 100
