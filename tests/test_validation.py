import pytest
import numpy as np
from pyreco.validation.cross_validation import cross_val
from pyreco.metrics.metrics import available_metrics

class TestCrossValidation:
    @pytest.fixture
    def sample_data(self):
        # Create a simple dataset for testing
        X = np.random.randn(100, 10)  # 100 samples, 10 features
        y = np.random.randn(100)      # 100 targets
        return X, y

    @pytest.fixture
    def mock_model(self):
        class MockModel:
            def __init__(self):
                self.fitted = False
                self.metric_value = 0.5

            def fit(self, X, y):
                self.fitted = True

            def evaluate(self, X, y, metric):
                return self.metric_value

        return MockModel()

    def test_cross_val_basic(self, mock_model, sample_data):
        X, y = sample_data
        n_splits = 5
        
        # Test with default metric (mse)
        fold_values, mean_val, std_val = cross_val(mock_model, X, y, n_splits)
        
        assert len(fold_values) == n_splits
        assert all(val == mock_model.metric_value for val in fold_values)
        assert mean_val == mock_model.metric_value
        assert std_val == 0.0  # All values are the same

    def test_cross_val_different_metrics(self, mock_model, sample_data):
        X, y = sample_data
        n_splits = 5
        
        # Test with different metrics
        for metric in available_metrics():
            fold_values, mean_val, std_val = cross_val(mock_model, X, y, n_splits, metric)
            assert len(fold_values) == n_splits
            assert mean_val == mock_model.metric_value

    def test_cross_val_different_splits(self, mock_model, sample_data):
        X, y = sample_data
        
        # Test with different numbers of splits
        for n_splits in [2, 3, 5, 10]:
            fold_values, mean_val, std_val = cross_val(mock_model, X, y, n_splits)
            assert len(fold_values) == n_splits

    def test_cross_val_small_dataset(self, mock_model):
        # Test with a very small dataset
        X = np.random.randn(5, 2)
        y = np.random.randn(5)
        
        # Should work with n_splits <= number of samples
        for n_splits in [2, 3, 5]:
            fold_values, mean_val, std_val = cross_val(mock_model, X, y, n_splits)
            assert len(fold_values) == n_splits

    def test_cross_val_invalid_splits(self, mock_model, sample_data):
        X, y = sample_data
        
        # Test with invalid number of splits
        with pytest.raises(ValueError):
            cross_val(mock_model, X, y, 0)  # Zero splits
        with pytest.raises(ValueError):
            cross_val(mock_model, X, y, -1)  # Negative splits
        with pytest.raises(ValueError):
            cross_val(mock_model, X, y, len(X) + 1)  # More splits than samples

    def test_cross_val_shuffling(self, mock_model, sample_data):
        X, y = sample_data
        n_splits = 5
        
        # Run cross-validation twice
        fold_values1, mean1, std1 = cross_val(mock_model, X, y, n_splits)
        fold_values2, mean2, std2 = cross_val(mock_model, X, y, n_splits)
        
        # The means should be the same (since we're using a mock model)
        assert mean1 == mean2
        assert std1 == std2
        
        # The fold values should be the same (since we're using a mock model)
        assert fold_values1 == fold_values2

    def test_cross_val_uneven_splits(self, mock_model):
        # Test with a dataset size that doesn't divide evenly by n_splits
        X = np.random.randn(7, 2)  # 7 samples
        y = np.random.randn(7)
        n_splits = 3
        
        fold_values, mean_val, std_val = cross_val(mock_model, X, y, n_splits)
        
        # Should have 3 folds with sizes 3, 2, 2
        assert len(fold_values) == n_splits 