"""
Tests for the CompositeSearch strategy.
"""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import MagicMock
from pyreco.reservoir_tuner.search.strategies import SearchStrategy
from pyreco.reservoir_tuner.search.composite import CompositeSearch

# Define DummyStrategy for testing
class DummyStrategy(SearchStrategy):
    def __init__(self, fixed_params: Dict[str, Any]):
        self.fixed_params = fixed_params
        self.observed_params = []
        self.observed_scores = []
    
    def suggest(self) -> Dict[str, Any]:
        return self.fixed_params
    
    def observe(self, params: Dict[str, Any], score: float) -> None:
        self.observed_params.append(params)
        self.observed_scores.append(score)

def test_composite_search_initialization():
    """Test initialization of CompositeSearch."""
    # Create dummy strategies
    strategy1 = DummyStrategy({"param1": 0.5})
    strategy2 = DummyStrategy({"param1": 1.0})
    
    # Test initialization with valid strategies
    composite = CompositeSearch([strategy1, strategy2])
    assert len(composite.strategies) == 2
    assert composite.exploration_factor == 2.0
    
    # Test initialization with custom exploration factor
    composite = CompositeSearch([strategy1, strategy2], exploration_factor=1.5)
    assert composite.exploration_factor == 1.5
    
    # Test initialization with empty strategy list
    with pytest.raises(ValueError):
        CompositeSearch([])

def test_composite_search_initial_exploration():
    """Test that each strategy is tried at least once initially."""
    # Create dummy strategies with different fixed parameters
    strategies = [
        DummyStrategy({"param1": 0.1}),
        DummyStrategy({"param1": 0.2}),
        DummyStrategy({"param1": 0.3})
    ]
    composite = CompositeSearch(strategies)
    
    # First three suggestions should use each strategy once
    params1 = composite.suggest()
    composite.observe(params1, 1.0)
    
    params2 = composite.suggest()
    composite.observe(params2, 1.0)
    
    params3 = composite.suggest()
    composite.observe(params3, 1.0)
    
    # Check that each strategy was used exactly once
    assert composite.arm_counts.tolist() == [1, 1, 1]

def test_composite_search_ucb_selection():
    """Test UCB-based strategy selection after initial exploration."""
    # Create dummy strategies
    strategies = [
        DummyStrategy({"param1": 0.1}),
        DummyStrategy({"param1": 0.2}),
        DummyStrategy({"param1": 0.3})
    ]
    composite = CompositeSearch(strategies, exploration_factor=1.0)
    
    # Initial exploration
    for _ in range(3):
        params = composite.suggest()
        composite.observe(params, 1.0)
    
    # Test that better performing strategies are selected more often
    # Strategy 0 gets high rewards
    for _ in range(5):
        params = composite.suggest()
        if params["param1"] == 0.1:  # Strategy 0
            composite.observe(params, 2.0)
        else:
            composite.observe(params, 0.5)
    
    # Strategy 0 should have been selected more often
    assert composite.arm_counts[0] > composite.arm_counts[1]
    assert composite.arm_counts[0] > composite.arm_counts[2]

def test_composite_search_observe():
    """Test observation handling and performance tracking."""
    # Create dummy strategies
    strategy1 = DummyStrategy({"param1": 0.5})
    strategy2 = DummyStrategy({"param1": 1.0})
    composite = CompositeSearch([strategy1, strategy2])
    
    # Test that observe() can't be called before suggest()
    with pytest.raises(RuntimeError):
        composite.observe({"param1": 0.5}, 1.0)
    
    # Test proper observation handling
    params = composite.suggest()
    composite.observe(params, 1.0)
    
    # Check that the correct strategy received the observation
    if params["param1"] == 0.5:
        assert len(strategy1.observed_params) == 1
        assert strategy1.observed_scores[0] == 1.0
    else:
        assert len(strategy2.observed_params) == 1
        assert strategy2.observed_scores[0] == 1.0

def test_composite_search_performance_tracking():
    """Test performance statistics tracking."""
    # Create dummy strategies
    strategies = [
        DummyStrategy({"param1": 0.1}),
        DummyStrategy({"param1": 0.2})
    ]
    composite = CompositeSearch(strategies)
    
    # Run some trials
    for _ in range(5):
        params = composite.suggest()
        if params["param1"] == 0.1:
            composite.observe(params, 2.0)
        else:
            composite.observe(params, 1.0)
    
    # Get performance statistics
    stats = composite.get_strategy_performance()
    
    # Check statistics format
    assert "counts" in stats
    assert "means" in stats
    assert "total_rewards" in stats
    
    # Check that statistics are consistent
    assert len(stats["counts"]) == 2
    assert len(stats["means"]) == 2
    assert len(stats["total_rewards"]) == 2
    assert sum(stats["counts"]) == 5 