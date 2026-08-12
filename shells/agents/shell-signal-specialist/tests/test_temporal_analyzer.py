"""
Tests for TemporalSignatureAnalyzer — Hermes OB1 Core
Verifies the pre-strike detection algorithm handles escalation patterns.
"""
import sys
import os
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from temporal_analyzer import TemporalSignatureAnalyzer


class TestTemporalSignatureAnalyzer:

    def setup_method(self):
        """Standard thresholds for testing."""
        self.analyzer = TemporalSignatureAnalyzer(
            acceleration_threshold=2.0,
            vertical_velocity_threshold=5.0
        )

    def test_insufficient_data_returns_zero(self):
        """Less than 3 data points should return 0.0 — can't compute acceleration."""
        data = np.array([[10, 1], [11, 1.2]])
        assert self.analyzer.analyze_escalation(data) == 0.0

    def test_empty_data_returns_zero(self):
        """Empty array should return 0.0."""
        data = np.array([]).reshape(0, 2)
        assert self.analyzer.analyze_escalation(data) == 0.0

    def test_normal_state_returns_zero(self):
        """Stable, non-escalating data should return 0.0."""
        data = np.array([[10, 1], [11, 1.2], [10.5, 1.1]])
        assert self.analyzer.analyze_escalation(data) == 0.0

    def test_escalation_detected(self):
        """Rapidly increasing intensity acceleration + high vertical velocity → high score."""
        # Intensities: 10 → 12 (v=2) → 17 (v=5, a=3 > threshold=2)
        # Vertical velocities: 1 → 6 → 7 (> threshold=5)
        data = np.array([[10, 1.0], [12, 6.0], [17, 7.0]])
        score = self.analyzer.analyze_escalation(data)
        assert score > 0.5
        assert score <= 1.0

    def test_acceleration_below_threshold_returns_zero(self):
        """Acceleration present but below threshold → no escalation."""
        # Intensities: 10 → 10.5 (v=0.5) → 11.2 (v=0.7, a=0.2 < threshold=2)
        data = np.array([[10, 1.0], [10.5, 1.0], [11.2, 1.0]])
        assert self.analyzer.analyze_escalation(data) == 0.0

    def test_vertical_velocity_below_threshold_returns_zero(self):
        """High acceleration but low vertical velocity → no escalation."""
        # Intensities: 10 → 15 (v=5) → 25 (v=10, a=5 > threshold=2)
        # Vertical velocities: 1 → 2 → 3 (< threshold=5)
        data = np.array([[10, 1.0], [15, 2.0], [25, 3.0]])
        assert self.analyzer.analyze_escalation(data) == 0.0

    def test_score_in_valid_range(self):
        """Escalation score should always be between 0 and 1."""
        data = np.array([[10, 1.0], [15, 6.0], [25, 8.0]])
        score = self.analyzer.analyze_escalation(data)
        assert 0.0 <= score <= 1.0

    def test_score_scales_with_severity(self):
        """More severe escalation should produce higher scores."""
        # Moderate escalation
        moderate = np.array([[10, 5.0], [13, 5.5], [18, 6.0]])
        # Extreme escalation
        extreme = np.array([[10, 10.0], [20, 15.0], [50, 20.0]])

        moderate_score = self.analyzer.analyze_escalation(moderate)
        extreme_score = self.analyzer.analyze_escalation(extreme)
        assert extreme_score >= moderate_score

    def test_custom_thresholds(self):
        """Analyzer with different thresholds should behave accordingly."""
        sensitive = TemporalSignatureAnalyzer(
            acceleration_threshold=0.1,
            vertical_velocity_threshold=0.1
        )
        # Mild data that wouldn't trigger the default analyzer
        data = np.array([[10, 0.5], [11, 0.6], [12.5, 0.7]])
        score = sensitive.analyze_escalation(data)
        # Should detect since thresholds are very low
        assert score > 0.0

    def test_negative_intensity_trend(self):
        """Decreasing intensity (cooling) should not trigger escalation."""
        data = np.array([[20, 1.0], [15, 1.0], [10, 1.0]])
        assert self.analyzer.analyze_escalation(data) == 0.0

    def test_exact_threshold_boundary(self):
        """Data exactly at threshold boundary — acceleration just over."""
        # Intensities: 0 → 2 (v=2) → 5 (v=3, a=1 < threshold=2) — no trigger
        # Actually let's compute: v=[2,3], a=[1], which is < 2 → no trigger
        data = np.array([[0, 6.0], [2, 6.0], [5, 6.0]])
        assert self.analyzer.analyze_escalation(data) == 0.0

    def test_minimum_three_points_required(self):
        """Exactly 3 points should work (minimum for acceleration calc)."""
        data = np.array([[10, 5.0], [15, 6.0], [25, 7.0]])
        # This should not crash and should return a valid score
        score = self.analyzer.analyze_escalation(data)
        assert 0.0 <= score <= 1.0

    def test_large_window(self):
        """Larger windows should work and use the latest data point."""
        # Build a window where escalation happens at the end
        # Need acceleration > 2.0 (strict >) at the latest point
        # Intensities: 5,5,5,5,5, 8,14,24 → v=[0,0,0,0,3,6,10] → a=[0,0,0,3,3,4] → latest a=4 > 2 ✓
        data = np.array([
            [5, 1], [5, 1], [5, 1], [5, 1], [5, 1],  # Stable baseline
            [8, 3], [14, 6], [24, 8]  # Escalation (last accel = 4 > threshold)
        ])
        score = self.analyzer.analyze_escalation(data)
        assert score > 0.0
