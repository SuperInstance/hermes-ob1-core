"""
Tests for PredictiveSonarEngine — Hermes OB1 Core
Verifies the sonar prediction engine handles edge cases correctly.
"""
import sys
import os
import numpy as np

# Add parent directory to path so we can import engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine import PredictiveSonarEngine


class TestPredictiveSonarEngine:
    """Test suite for the predictive sonar engine."""

    def setup_method(self):
        """Fresh engine for each test."""
        self.engine = PredictiveSonarEngine(dt=0.1)

    def test_initial_intensity_derivative_is_zero(self):
        """First reading should return 0.0 derivative — no history yet."""
        result = self.engine.calculate_intensity_derivative(10.0)
        assert result == 0.0

    def test_intensity_derivative_single_step(self):
        """After two readings, derivative should be (I2-I1)/dt."""
        self.engine.calculate_intensity_derivative(10.0)
        result = self.engine.calculate_intensity_derivative(15.0)
        expected = (15.0 - 10.0) / 0.1
        assert abs(result - expected) < 1e-10

    def test_intensity_derivative_decreasing(self):
        """Decreasing intensity should give negative derivative."""
        self.engine.calculate_intensity_derivative(20.0)
        result = self.engine.calculate_intensity_derivative(10.0)
        assert result < 0

    def test_intensity_derivative_constant(self):
        """Constant intensity should give zero derivative."""
        self.engine.calculate_intensity_derivative(5.0)
        result = self.engine.calculate_intensity_derivative(5.0)
        assert result == 0.0

    def test_intensity_history_depth_limit(self):
        """History should not exceed configured depth."""
        for i in range(20):
            self.engine.calculate_intensity_derivative(float(i))
        assert len(self.engine.intensity_history) <= self.engine.history_depth

    def test_trajectory_vector_first_reading(self):
        """First position reading should return zero velocity and confidence 1.0."""
        vel, conf = self.engine.calculate_trajectory_vector([0.0, 0.0], 0.0)
        assert np.allclose(vel, [0.0, 0.0])
        assert conf == 1.0

    def test_trajectory_vector_basic_motion(self):
        """Two readings should give correct velocity."""
        self.engine.calculate_trajectory_vector([0.0, 0.0], 0.0)
        vel, conf = self.engine.calculate_trajectory_vector([1.0, 0.0], 5.0)
        # dx = 1.0, dt = 0.1, so velocity = 10.0 in x
        assert abs(vel[0] - 10.0) < 1e-10
        assert abs(vel[1] - 0.0) < 1e-10

    def test_trajectory_confidence_scales_with_derivative(self):
        """Higher intensity derivative should give higher confidence."""
        self.engine.calculate_trajectory_vector([0.0, 0.0], 0.0)
        _, low_conf = self.engine.calculate_trajectory_vector([0.1, 0.0], 1.0)

        self.engine2 = PredictiveSonarEngine(dt=0.1)
        self.engine2.calculate_trajectory_vector([0.0, 0.0], 0.0)
        _, high_conf = self.engine2.calculate_trajectory_vector([0.1, 0.0], 100.0)

        assert high_conf >= low_conf

    def test_position_history_depth_limit(self):
        """Position history should respect depth limit."""
        for i in range(20):
            self.engine.calculate_trajectory_vector([float(i), 0.0], 1.0)
        assert len(self.engine.position_history) <= self.engine.history_depth

    def test_predict_next_position_linear(self):
        """Prediction should be linear extrapolation from current pos + velocity * dt."""
        pos = np.array([5.0, 3.0])
        vel = np.array([2.0, -1.0])
        predicted = self.engine.predict_next_position(pos, vel, dt_future=2.0)
        expected = pos + vel * 2.0
        assert np.allclose(predicted, expected)

    def test_predict_next_position_zero_velocity(self):
        """Zero velocity means prediction equals current position."""
        pos = np.array([7.0, 4.0])
        predicted = self.engine.predict_next_position(pos, np.array([0.0, 0.0]), dt_future=10.0)
        assert np.allclose(predicted, pos)

    def test_predict_next_position_negative_dt(self):
        """Negative dt should extrapolate backward (mathematically valid)."""
        pos = np.array([0.0, 0.0])
        vel = np.array([1.0, 1.0])
        predicted = self.engine.predict_next_position(pos, vel, dt_future=-1.0)
        assert np.allclose(predicted, [-1.0, -1.0])

    def test_custom_dt(self):
        """Custom dt should affect derivative calculations."""
        engine = PredictiveSonarEngine(dt=0.5)
        engine.calculate_intensity_derivative(10.0)
        result = engine.calculate_intensity_derivative(20.0)
        expected = (20.0 - 10.0) / 0.5
        assert abs(result - expected) < 1e-10

    def test_custom_history_depth(self):
        """Custom history depth should be respected."""
        engine = PredictiveSonarEngine(dt=0.1)
        engine.history_depth = 3
        for i in range(10):
            engine.calculate_intensity_derivative(float(i))
        assert len(engine.intensity_history) <= 3

    def test_2d_trajectory_diagonal(self):
        """Diagonal movement should produce diagonal velocity."""
        self.engine.calculate_trajectory_vector([0.0, 0.0], 0.0)
        vel, _ = self.engine.calculate_trajectory_vector([1.0, 1.0], 5.0)
        # Both components equal
        assert abs(vel[0] - vel[1]) < 1e-10
