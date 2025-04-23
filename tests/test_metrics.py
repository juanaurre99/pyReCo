import pytest
import numpy as np
from pyreco.metrics.metrics import (
    mse,
    mae,
    r2,
    available_metrics,
    assign_metric
)

class TestMetrics:
    @pytest.fixture
    def sample_data(self):
        # Create sample data for testing
        y_true = np.array([
            [[1.0, 2.0], [3.0, 4.0]],  # First batch
            [[5.0, 6.0], [7.0, 8.0]]   # Second batch
        ])
        y_pred = np.array([
            [[1.1, 2.2], [2.8, 4.2]],  # First batch
            [[4.8, 5.8], [7.2, 7.8]]   # Second batch
        ])
        return y_true, y_pred

    def test_mse(self, sample_data):
        y_true, y_pred = sample_data
        
        # Test MSE calculation
        error = mse(y_true, y_pred)
        
        # Calculate expected MSE manually
        diff = y_true - y_pred
        expected_mse = np.mean(diff ** 2)
        
        assert np.isclose(error, expected_mse)
        
        # Test with perfect prediction
        assert mse(y_true, y_true) == 0.0
        
        # Test with constant offset
        offset = 1.0
        error_with_offset = mse(y_true, y_true + offset)
        assert np.isclose(error_with_offset, offset ** 2)

    def test_mae(self, sample_data):
        y_true, y_pred = sample_data
        
        # Test MAE calculation
        error = mae(y_true, y_pred)
        
        # Calculate expected MAE manually
        diff = np.abs(y_true - y_pred)
        expected_mae = np.mean(diff)
        
        assert np.isclose(error, expected_mae)
        
        # Test with perfect prediction
        assert mae(y_true, y_true) == 0.0
        
        # Test with constant offset
        offset = 1.0
        error_with_offset = mae(y_true, y_true + offset)
        assert np.isclose(error_with_offset, abs(offset))

    def test_r2(self, sample_data):
        y_true, y_pred = sample_data
        
        # Test R² calculation
        score = r2(y_true, y_pred)
        
        # Score should be between 0 and 1 for reasonable predictions
        assert 0 <= score <= 1
        
        # Test with perfect prediction
        assert np.isclose(r2(y_true, y_true), 1.0)
        
        # Test with mean prediction (should give R² = 0)
        y_mean = np.full_like(y_true, np.mean(y_true))
        assert np.isclose(r2(y_true, y_mean), 0.0)

    def test_available_metrics(self):
        metrics = available_metrics()
        
        # Check if all expected metrics are available
        expected_metrics = [
            "mse", "mean_squared_error",
            "mae", "mean_absolute_error",
            "r2", "r2_score"
        ]
        
        assert set(metrics) == set(expected_metrics)
        assert len(metrics) == len(expected_metrics)

    def test_assign_metric(self):
        # Test assigning MSE
        metric_fn = assign_metric("mse")
        assert metric_fn == mse
        
        metric_fn = assign_metric("mean_squared_error")
        assert metric_fn == mse
        
        # Test assigning MAE
        metric_fn = assign_metric("mae")
        assert metric_fn == mae
        
        metric_fn = assign_metric("mean_absolute_error")
        assert metric_fn == mae
        
        # Test assigning R²
        metric_fn = assign_metric("r2")
        assert metric_fn == r2
        
        metric_fn = assign_metric("r2_score")
        assert metric_fn == r2
        
        # Test invalid metric
        with pytest.raises(ValueError):
            assign_metric("invalid_metric")

    def test_metric_shapes(self):
        # Test metrics with different input shapes
        shapes = [
            (1, 1, 1),    # Single value
            (2, 3, 4),    # Multiple batches, timesteps, features
            (5, 1, 2),    # Multiple batches, single timestep
            (1, 10, 1)    # Single batch, multiple timesteps
        ]
        
        for shape in shapes:
            y_true = np.random.randn(*shape)
            y_pred = np.random.randn(*shape)
            
            # All metrics should handle these shapes
            mse_val = mse(y_true, y_pred)
            mae_val = mae(y_true, y_pred)
            r2_val = r2(y_true, y_pred)
            
            assert isinstance(mse_val, float)
            assert isinstance(mae_val, float)
            assert isinstance(r2_val, float)

    def test_metric_edge_cases(self):
        # Test with zero arrays
        zeros = np.zeros((2, 2, 2))
        assert mse(zeros, zeros) == 0.0
        assert mae(zeros, zeros) == 0.0
        assert np.isclose(r2(zeros, zeros), 1.0)
        
        # Test with NaN handling
        y_true = np.array([1.0, 2.0, np.nan])
        y_pred = np.array([1.1, 2.2, np.nan])
        
        # NaN values should propagate to the result
        assert np.isnan(mse(y_true, y_pred))
        assert np.isnan(mae(y_true, y_pred))
        assert np.isnan(r2(y_true, y_pred))
        
        # Test with different sizes
        y_true = np.random.randn(2, 3, 4)
        y_pred = np.random.randn(2, 3, 5)
        
        with pytest.raises(ValueError):  # Should raise error for size mismatch
            mse(y_true, y_pred) 