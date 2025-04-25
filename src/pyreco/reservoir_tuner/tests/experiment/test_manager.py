"""
Tests for the ExperimentManager class.
"""

import pytest
import numpy as np
from pathlib import Path
import yaml
import json
import tempfile

from pyreco.reservoir_tuner.experiment.manager import ExperimentManager
from pyreco.reservoir_tuner.experiment.config import ExperimentConfig
from pyreco.reservoir_tuner.search.random import RandomSearch
from pyreco.reservoir_tuner.search.bayesian import BayesianSearch
from pyreco.reservoir_tuner.search.composite import CompositeSearch


@pytest.fixture
def valid_config():
    return {
        'task': {
            'type': 'classification',
            'input_shape': [10, 1],  # [sequence_length, features]
            'output_shape': [1],
            'sequence_length': 10
        },
        'search_space': {
            'reservoir_size': {'type': 'discrete', 'range': [50, 200]},
            'spectral_radius': {'type': 'float', 'range': [0.1, 2.0]},
            'input_scaling': {'type': 'float', 'range': [0.1, 2.0]},
            'leaking_rate': {'type': 'float', 'range': [0.1, 1.0]},
            'bias_scaling': {'type': 'float', 'range': [0.0, 2.0]},
            'noise_level': {'type': 'float', 'range': [0.0, 0.1]}
        },
        'search_strategy': {
            'type': 'random',
            'seed': 42
        },
        'max_trials': 10
    }


@pytest.fixture
def valid_config_dict():
    """Fixture providing a valid experiment configuration dictionary."""
    return {
        "experiment": {
            "name": "test_experiment",
            "description": "Test experiment",
            "seed": 42
        },
        "task": {
            "type": "vector_to_vector",
            "name": "sine_prediction",
            "input_dim": 10,
            "output_dim": 10,
            "sequence_length": 10,
            "train_ratio": 0.8,
            "validation_ratio": 0.1
        },
        "model": {
            "type": "reservoir",
            "input_layer": {
                "size": 10,
                "type": "dense"
            },
            "reservoir_layer": {
                "size": 100,
                "type": "reservoir",
                "connectivity": 0.1
            },
            "output_layer": {
                "size": 10,
                "type": "dense"
            }
        },
        "search_space": {
            "spectral_radius": {
                "type": "float",
                "range": [0.1, 2.0]
            },
            "sparsity": {
                "type": "float",
                "range": [0.1, 0.9]
            }
        },
        "optimization": {
            "strategy": "random",
            "max_trials": 10,
            "strategies": [
                {
                    "type": "random",
                    "weight": 1.0
                }
            ]
        },
        "metrics": {
            "primary": "mse",
            "secondary": ["mae", "rmse"],
            "resource": ["time", "memory"]
        }
    }


@pytest.fixture
def yaml_config_file(valid_config_dict):
    """Fixture creating a temporary YAML config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(valid_config_dict, f)
        return f.name


@pytest.fixture
def json_config_file(valid_config_dict):
    """Fixture creating a temporary JSON config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(valid_config_dict, f)
        return f.name


def test_experiment_manager_initialization(valid_config_dict):
    """Test that ExperimentManager initializes correctly with valid config."""
    manager = ExperimentManager(valid_config_dict)
    assert isinstance(manager.config, ExperimentConfig)
    assert isinstance(manager.search_strategy, RandomSearch)
    
    # Test task data creation
    assert len(manager.task_data) == 4  # X_train, X_test, y_train, y_test
    assert all(isinstance(arr, np.ndarray) for arr in manager.task_data)
    assert manager.task_data[0].shape[1] == valid_config_dict["task"]["input_dim"]


def test_config_loading_from_yaml(yaml_config_file):
    """Test loading configuration from YAML file."""
    manager = ExperimentManager(yaml_config_file)
    assert isinstance(manager.config, ExperimentConfig)
    assert manager.config.name == "test_experiment"


def test_config_loading_from_json(json_config_file):
    """Test loading configuration from JSON file."""
    manager = ExperimentManager(json_config_file)
    assert isinstance(manager.config, ExperimentConfig)
    assert manager.config.name == "test_experiment"


def test_get_next_trial_config(valid_config_dict):
    """Test that get_next_trial_config returns valid configuration."""
    manager = ExperimentManager(valid_config_dict)
    trial_config = manager.get_next_trial_config()
    
    # Check structure
    assert "model" in trial_config
    assert "task" in trial_config
    assert "hyperparameters" in trial_config
    assert "data" in trial_config
    
    # Check data
    assert isinstance(trial_config["data"]["X_train"], np.ndarray)
    assert isinstance(trial_config["data"]["X_test"], np.ndarray)
    assert isinstance(trial_config["data"]["y_train"], np.ndarray)
    assert isinstance(trial_config["data"]["y_test"], np.ndarray)
    
    # Check hyperparameters
    assert "spectral_radius" in trial_config["hyperparameters"]
    assert "sparsity" in trial_config["hyperparameters"]
    assert 0.1 <= trial_config["hyperparameters"]["spectral_radius"] <= 2.0
    assert 0.1 <= trial_config["hyperparameters"]["sparsity"] <= 0.9


def test_update_search_strategy(valid_config_dict):
    """Test updating the search strategy with trial results."""
    manager = ExperimentManager(valid_config_dict)
    params = {"spectral_radius": 1.0, "sparsity": 0.5}
    score = 0.1
    
    # This should not raise any errors
    manager.update_search_strategy(params, score)


def test_invalid_config():
    """Test that ExperimentManager raises appropriate errors for invalid configs."""
    # Missing required fields
    with pytest.raises(ValueError):
        ExperimentManager({})
    
    # Invalid task type
    invalid_config = {
        "experiment": {"name": "test", "description": "test", "seed": 42},
        "task": {
            "type": "invalid_type",
            "input_dim": 10,
            "sequence_length": 1,
            "train_ratio": 0.8,
            "validation_ratio": 0.1
        },
        "model": {
            "type": "reservoir",
            "input_layer": {"size": 10},
            "reservoir_layer": {"size": 100},
            "output_layer": {"size": 10}
        },
        "search_space": {},
        "optimization": {
            "strategy": "random",
            "max_trials": 10,
            "strategies": [{"type": "random", "weight": 1.0}]
        },
        "metrics": {
            "primary": "mse",
            "secondary": [],
            "resource": []
        }
    }
    with pytest.raises(ValueError):
        ExperimentManager(invalid_config)


def test_composite_strategy_creation():
    """Test creation of composite search strategy."""
    config = {
        "experiment": {"name": "test", "description": "test", "seed": 42},
        "task": {
            "type": "vector_to_vector",
            "name": "sine_prediction",
            "input_dim": 10,
            "output_dim": 10,
            "sequence_length": 10,
            "train_ratio": 0.8,
            "validation_ratio": 0.1
        },
        "model": {
            "type": "reservoir",
            "input_layer": {"size": 10},
            "reservoir_layer": {"size": 100},
            "output_layer": {"size": 10}
        },
        "search_space": {
            "spectral_radius": {"type": "float", "range": [0.1, 2.0]},
            "sparsity": {"type": "float", "range": [0.1, 0.9]}
        },
        "optimization": {
            "strategy": "composite",
            "max_trials": 10,
            "strategies": [
                {"type": "random", "weight": 0.5},
                {"type": "bayesian", "weight": 0.5}
            ]
        },
        "metrics": {
            "primary": "mse",
            "secondary": [],
            "resource": []
        }
    }
    
    manager = ExperimentManager(config)
    assert isinstance(manager.search_strategy, CompositeSearch)
    assert len(manager.search_strategy.strategies) == 2
    assert isinstance(manager.search_strategy.strategies[0], RandomSearch)
    assert isinstance(manager.search_strategy.strategies[1], BayesianSearch)


def test_properties(valid_config_dict):
    """Test access to experiment properties."""
    manager = ExperimentManager(valid_config_dict)
    
    assert manager.max_trials == 10
    assert manager.experiment_name == "test_experiment"
    assert manager.metrics_config["primary"] == "mse"
    assert "mae" in manager.metrics_config["secondary"]
    assert "time" in manager.metrics_config["resource"] 