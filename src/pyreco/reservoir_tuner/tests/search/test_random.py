"""
Test cases for the RandomSearch strategy.
"""
import pytest
import numpy as np
from pyreco.reservoir_tuner.search.random import RandomSearch

@pytest.fixture
def parameter_space():
    return {
        'learning_rate': {
            'type': 'continuous',
            'min': 0.0001,
            'max': 0.1
        },
        'neurons': {
            'type': 'discrete',
            'min': 50,
            'max': 200
        },
        'activation': {
            'type': 'categorical',
            'values': ['tanh', 'relu', 'sigmoid']
        }
    }

def test_random_search_initialization(parameter_space):
    # Test initialization with default seed
    rs = RandomSearch(parameter_space)
    assert rs.parameter_space == parameter_space
    
    # Test initialization with specific seed
    rs_seeded = RandomSearch(parameter_space, seed=42)
    assert rs_seeded.parameter_space == parameter_space

def test_random_search_reproducibility(parameter_space):
    # Test that same seed produces same suggestions
    rs1 = RandomSearch(parameter_space, seed=42)
    rs2 = RandomSearch(parameter_space, seed=42)
    
    suggestion1 = rs1.suggest()
    suggestion2 = rs2.suggest()
    
    assert suggestion1 == suggestion2

def test_random_search_suggestions(parameter_space):
    rs = RandomSearch(parameter_space, seed=42)
    
    # Test multiple suggestions
    for _ in range(10):
        suggestion = rs.suggest()
        
        # Check learning rate bounds
        assert 0.0001 <= suggestion['learning_rate'] <= 0.1
        
        # Check neurons bounds and type
        assert 50 <= suggestion['neurons'] <= 200
        assert isinstance(suggestion['neurons'], (int, np.integer))
        
        # Check activation is from allowed values
        assert suggestion['activation'] in ['tanh', 'relu', 'sigmoid']

def test_random_search_observe(parameter_space):
    rs = RandomSearch(parameter_space)
    params = rs.suggest()
    score = 0.95
    
    # Observe should not raise any errors
    rs.observe(params, score)

def test_invalid_parameter_space():
    invalid_space = {
        'param': {
            'type': 'invalid_type',
            'min': 0,
            'max': 1
        }
    }
    
    rs = RandomSearch(invalid_space)
    with pytest.raises(ValueError, match="Unknown parameter type: invalid_type"):
        rs.suggest() 