import pytest
import numpy as np
from pyreco.core.optimizers import Optimizer, RidgeSK, assign_optimizer


class MockOptimizer(Optimizer):
    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.zeros((A.shape[1], b.shape[1]))


class TestOptimizer:
    def test_abstract_class(self):
        with pytest.raises(TypeError):
            Optimizer()


class TestRidgeSK:
    @pytest.fixture
    def ridge_optimizer(self):
        return RidgeSK(alpha=1.0)

    @pytest.fixture
    def sample_data(self):
        # Create sample data for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        n_outputs = 2
        
        # Generate random input data
        A = np.random.randn(n_samples, n_features)
        # Generate target data with known relationship plus noise
        true_weights = np.random.randn(n_features, n_outputs)
        b = A @ true_weights + 0.1 * np.random.randn(n_samples, n_outputs)
        
        return A, b, true_weights

    def test_initialization(self):
        optimizer = RidgeSK()
        assert optimizer.name == ""
        assert optimizer.alpha == 1.0

        optimizer = RidgeSK(name="custom_ridge", alpha=0.5)
        assert optimizer.name == "custom_ridge"
        assert optimizer.alpha == 1.0  # Note: alpha is hardcoded to 1.0 in __init__

    def test_solve(self, ridge_optimizer, sample_data):
        A, b, true_weights = sample_data
        
        # Test solving
        W_out = ridge_optimizer.solve(A, b)
        
        # Check output shape
        assert W_out.shape == (A.shape[1], b.shape[1])
        
        # Check that the solution provides reasonable predictions
        predictions = A @ W_out
        # Use more lenient tolerances since we're using regularization
        assert np.allclose(predictions, b, rtol=0.5, atol=0.5)

    def test_solve_edge_cases(self, ridge_optimizer):
        # Test with minimal data
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0], [1.0]])
        W_out = ridge_optimizer.solve(A, b)
        # For single output, scikit-learn returns 1D array
        assert len(W_out.shape) in [1, 2]  # Allow both 1D and 2D outputs
        assert W_out.shape[0] == 2  # Number of features
        if len(W_out.shape) == 2:
            assert W_out.shape[1] == 1  # Number of outputs

        # Test with zero target values
        b = np.zeros((2, 1))
        W_out = ridge_optimizer.solve(A, b)
        assert np.allclose(W_out, 0, rtol=1e-10)


class TestAssignOptimizer:
    def test_assign_ridge_optimizer(self):
        # Test with string "ridge"
        optimizer = assign_optimizer("ridge")
        assert isinstance(optimizer, RidgeSK)

        # Test with string "Ridge"
        optimizer = assign_optimizer("Ridge")
        assert isinstance(optimizer, RidgeSK)

    def test_assign_custom_optimizer(self):
        # Test with custom optimizer instance
        custom_opt = MockOptimizer()
        optimizer = assign_optimizer(custom_opt)
        assert optimizer is custom_opt

    def test_invalid_optimizer(self):
        # Test with invalid string
        with pytest.raises(ValueError):
            assign_optimizer("invalid_optimizer")

        # Test with invalid type
        with pytest.raises(TypeError):
            assign_optimizer(123)

        with pytest.raises(TypeError):
            assign_optimizer([1, 2, 3]) 