"""
Tests for the SearchStrategy interface and its implementations.
"""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import patch, MagicMock
from pyreco.reservoir_tuner.search.strategies import SearchStrategy, RandomSearch, BayesianSearch

# Define DummyStrategy at module level
class DummyStrategy(SearchStrategy):
    """
    A dummy implementation of SearchStrategy for testing.
    """
    def suggest(self) -> Dict[str, Any]:
        return {"param1": 0.5}
    
    def observe(self, params: Dict[str, Any], score: float) -> None:
        pass

def test_search_strategy_interface():
    """
    Test that SearchStrategy is abstract and that a concrete implementation works.
    """
    # Test that SearchStrategy is abstract
    with pytest.raises(TypeError):
        SearchStrategy()

    # Test that a subclass must implement abstract methods
    class IncompleteStrategy(SearchStrategy):
        pass

    with pytest.raises(TypeError):
        IncompleteStrategy()

    # Test that a complete subclass works
    strategy = DummyStrategy()
    assert strategy.suggest() == {"param1": 0.5}
    strategy.observe({"param1": 0.5}, 0.8)

def test_random_search_initialization():
    # Test initialization with valid parameter ranges
    param_ranges = {
        "param1": (0.0, 1.0),
        "param2": (-1.0, 1.0),
        "param3": (0, 10)
    }
    strategy = RandomSearch(param_ranges)
    assert strategy.param_ranges == param_ranges

    # Test initialization with invalid parameter ranges
    with pytest.raises(ValueError):
        RandomSearch({"param1": (1.0, 0.0)})  # min > max

    with pytest.raises(ValueError):
        RandomSearch({"param1": (0.0, 1.0, 2.0)})  # too many values

def test_random_search_suggest():
    # Test 1D parameter space sampling
    param_ranges = {"param1": (0.0, 1.0)}
    strategy = RandomSearch(param_ranges)
    
    # Test multiple samples
    samples = [strategy.suggest() for _ in range(1000)]
    values = [s["param1"] for s in samples]
    
    # Check range
    assert all(0.0 <= v <= 1.0 for v in values)
    
    # Check distribution (roughly uniform)
    hist, _ = np.histogram(values, bins=10)
    assert np.all(hist > 50)  # Each bin should have at least 50 samples

def test_random_search_multiple_params():
    # Test multiple parameter sampling
    param_ranges = {
        "param1": (0.0, 1.0),
        "param2": (-1.0, 1.0),
        "param3": (0, 10)
    }
    strategy = RandomSearch(param_ranges)
    
    # Test multiple samples
    samples = [strategy.suggest() for _ in range(100)]
    
    for sample in samples:
        # Check all parameters are present
        assert set(sample.keys()) == set(param_ranges.keys())
        
        # Check each parameter is within its range
        assert 0.0 <= sample["param1"] <= 1.0
        assert -1.0 <= sample["param2"] <= 1.0
        assert 0 <= sample["param3"] <= 10

def test_random_search_observe():
    # Test that observe method works (should do nothing for random search)
    param_ranges = {"param1": (0.0, 1.0)}
    strategy = RandomSearch(param_ranges)
    
    # Should not raise any errors
    strategy.observe({"param1": 0.5}, 0.8)
    strategy.observe({"param1": 0.3}, 0.6)

@patch('pyreco.reservoir_tuner.search.strategies.SKOPT_AVAILABLE', True)
def test_bayesian_search_initialization():
    """Test BayesianSearch initialization with and without scikit-optimize."""
    param_ranges = {
        "param1": (0.0, 1.0),
        "param2": (-1.0, 1.0)
    }
    
    # Create a mock optimizer
    mock_optimizer = MagicMock()
    mock_skopt = MagicMock()
    mock_skopt.Optimizer = MagicMock(return_value=mock_optimizer)
    
    # Test initialization with scikit-optimize available
    with patch('pyreco.reservoir_tuner.search.strategies.skopt', mock_skopt):
        strategy = BayesianSearch(param_ranges)
        assert strategy.param_ranges == param_ranges
        assert strategy.optimizer is not None
        assert strategy.fallback_strategy is None
        
        # Verify optimizer was created with correct parameters
        mock_skopt.Optimizer.assert_called_once()
        call_args = mock_skopt.Optimizer.call_args[1]
        assert call_args['base_estimator'] == "GP"
        assert call_args['acq_func'] == "EI"
        assert call_args['n_initial_points'] == 5

@patch('pyreco.reservoir_tuner.search.strategies.SKOPT_AVAILABLE', False)
def test_bayesian_search_fallback():
    """Test BayesianSearch fallback to RandomSearch when scikit-optimize is unavailable."""
    param_ranges = {
        "param1": (0.0, 1.0),
        "param2": (-1.0, 1.0)
    }
    
    strategy = BayesianSearch(param_ranges)
    assert isinstance(strategy.fallback_strategy, RandomSearch)
    assert strategy.optimizer is None
    
    # Test suggest
    params = strategy.suggest()
    assert set(params.keys()) == {"param1", "param2"}
    assert 0.0 <= params["param1"] <= 1.0
    assert -1.0 <= params["param2"] <= 1.0
    
    # Test observe (should not raise any errors)
    strategy.observe(params, 0.8)

@patch('pyreco.reservoir_tuner.search.strategies.SKOPT_AVAILABLE', True)
def test_bayesian_search_suggest():
    """Test BayesianSearch suggest method with scikit-optimize."""
    param_ranges = {
        "param1": (0.0, 1.0),
        "param2": (-1.0, 1.0)
    }
    
    # Create a mock optimizer that returns a fixed point
    mock_optimizer = MagicMock()
    mock_optimizer.ask.return_value = [0.5, 0.0]
    mock_skopt = MagicMock()
    mock_skopt.Optimizer = MagicMock(return_value=mock_optimizer)
    
    with patch('pyreco.reservoir_tuner.search.strategies.skopt', mock_skopt):
        strategy = BayesianSearch(param_ranges)
        params = strategy.suggest()
        
        assert set(params.keys()) == {"param1", "param2"}
        assert params["param1"] == 0.5
        assert params["param2"] == 0.0
        mock_optimizer.ask.assert_called_once()

@patch('pyreco.reservoir_tuner.search.strategies.SKOPT_AVAILABLE', True)
def test_bayesian_search_observe():
    """Test BayesianSearch observe method with scikit-optimize."""
    param_ranges = {
        "param1": (0.0, 1.0),
        "param2": (-1.0, 1.0)
    }
    
    # Create a mock optimizer
    mock_optimizer = MagicMock()
    mock_skopt = MagicMock()
    mock_skopt.Optimizer = MagicMock(return_value=mock_optimizer)
    
    with patch('pyreco.reservoir_tuner.search.strategies.skopt', mock_skopt):
        strategy = BayesianSearch(param_ranges)
        strategy.observe({"param1": 0.5, "param2": 0.0}, 0.8)
        
        mock_optimizer.tell.assert_called_once_with(
            [0.5, 0.0], -0.8  # Note: minimizing negative score is equivalent to maximizing score
        ) 