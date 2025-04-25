"""
End-to-end tests for the hyperparameter tuning framework.
Tests the complete pipeline from configuration to result analysis.
"""

import os
import time
import pytest
import numpy as np
import json
import yaml
from pathlib import Path

from pyreco.reservoir_tuner.experiment.manager import ExperimentManager
from pyreco.reservoir_tuner.search.strategies import RandomSearch, BayesianSearch
from pyreco.reservoir_tuner.search.composite import CompositeSearch
from pyreco.reservoir_tuner.metrics.aggregator import ResultAggregator

@pytest.fixture
def sample_task_data():
    """Generate sample task data for testing."""
    # Generate simple sine wave data
    t = np.linspace(0, 2*np.pi, 100)
    X = np.sin(t).reshape(-1, 1)
    y = np.cos(t).reshape(-1, 1)
    
    # Split into train/test
    train_size = int(0.8 * len(X))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Reshape to 3D (n_batch, n_timesteps, n_features)
    sequence_length = 10
    n_features = 1
    
    # Create sequences for X_train
    n_train_sequences = len(X_train) - sequence_length + 1
    X_train_3d = np.zeros((n_train_sequences, sequence_length, n_features))
    y_train_3d = np.zeros((n_train_sequences, sequence_length, n_features))
    for i in range(n_train_sequences):
        X_train_3d[i] = X_train[i:i+sequence_length].reshape(sequence_length, n_features)
        y_train_3d[i] = y_train[i:i+sequence_length].reshape(sequence_length, n_features)
    
    # Create sequences for X_test
    n_test_sequences = len(X_test) - sequence_length + 1
    X_test_3d = np.zeros((n_test_sequences, sequence_length, n_features))
    y_test_3d = np.zeros((n_test_sequences, sequence_length, n_features))
    for i in range(n_test_sequences):
        X_test_3d[i] = X_test[i:i+sequence_length].reshape(sequence_length, n_features)
        y_test_3d[i] = y_test[i:i+sequence_length].reshape(sequence_length, n_features)
    
    return {
        'train_data': (X_train_3d, y_train_3d),
        'val_data': (X_test_3d, y_test_3d),
        'input_shape': [sequence_length, n_features],
        'output_shape': [sequence_length, n_features]
    }

@pytest.fixture
def valid_config(tmp_path):
    """Create a valid configuration for testing."""
    config = {
        "experiment": {
            "name": "test_experiment",
            "description": "Test experiment configuration",
            "seed": 42,
            "output_dir": str(tmp_path)
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
                "type": "dense",
                "size": 10,
                "activation": "linear"
            },
            "reservoir_layer": {
                "type": "reservoir",
                "size": 100,
                "activation": "tanh",
                "connectivity": 0.1
            },
            "output_layer": {
                "type": "dense",
                "size": 10,
                "activation": "linear"
            }
        },
        "search_space": {
            "nodes": {"type": "int", "range": [50, 200]},
            "density": {"type": "float", "range": [0.1, 0.9]},
            "activation": {"type": "categorical", "values": ["tanh"]},
            "leakage_rate": {"type": "float", "range": [0.1, 1.0]},
            "alpha": {"type": "float", "range": [0.1, 2.0]}
        },
        "optimization": {
            "strategy": "composite",
            "max_trials": 10,
            "strategies": [
                {"type": "random", "weight": 0.5, "exploration_factor": 2.0},
                {"type": "bayesian", "weight": 0.5, "exploration_factor": 2.0}
            ]
        },
        "metrics": {
            "primary": "mse",
            "secondary": ["mae", "rmse"],
            "resource": ["time", "memory"]
        }
    }
    
    # Save to temporary file
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    return str(config_file)

def test_smoke_test(valid_config, sample_task_data, tmp_path):
    """Test that the end-to-end workflow runs without errors."""
    # Create experiment manager
    manager = ExperimentManager(valid_config)
    
    # Set task data
    manager.set_task_data(
        train_data=sample_task_data['train_data'],
        val_data=sample_task_data['val_data']
    )
    
    # Run experiment
    results = manager.run()
    
    # Basic assertions
    assert isinstance(results, dict)
    assert "trials" in results
    assert len(results["trials"]) > 0
    
    # Check trial structure
    trial = results["trials"][0]
    assert "config" in trial
    assert "metrics" in trial
    assert "mse" in trial["metrics"]
    
    # Check output files
    assert os.path.exists(os.path.join(tmp_path, "results.json"))
    assert os.path.exists(os.path.join(tmp_path, "best_model.pkl"))

def test_integration_random_search(valid_config, sample_task_data):
    """
    Integration test for RandomSearch strategy.
    """
    # Modify config to use only RandomSearch
    with open(valid_config) as f:
        config = yaml.safe_load(f)
    
    config['search_strategy'] = {'type': 'random'}
    
    # Initialize manager with modified config
    manager = ExperimentManager.from_dict(config)
    manager.set_task_data(
        train_data=sample_task_data['train_data'],
        val_data=sample_task_data['val_data']
    )
    
    # Run trials
    results = []
    for _ in range(3):
        config = manager.get_next_trial_config()
        result = manager.run_trial(config)
        results.append(result)
        manager.update_search_strategy(config['hyperparameters'], result['metrics']['mse'])
    
    # Verify results
    assert len(results) == 3
    assert all('hyperparameters' in r for r in results)
    assert all(set(config['search_space'].keys()) == set(r['hyperparameters'].keys()) 
              for r in results)

def test_integration_bayesian_search(valid_config, sample_task_data):
    """
    Integration test for BayesianSearch strategy.
    """
    # Modify config to use only BayesianSearch
    with open(valid_config) as f:
        config = yaml.safe_load(f)
    
    config['search_strategy'] = {'type': 'bayesian'}
    
    # Initialize manager with modified config
    manager = ExperimentManager.from_dict(config)
    manager.set_task_data(
        train_data=sample_task_data['train_data'],
        val_data=sample_task_data['val_data']
    )
    
    # Run trials
    results = []
    for _ in range(3):
        config = manager.get_next_trial_config()
        result = manager.run_trial(config)
        results.append(result)
        manager.update_search_strategy(config['hyperparameters'], result['metrics']['mse'])
    
    # Verify results show improvement
    mse_scores = [r['metrics']['mse'] for r in results]
    assert min(mse_scores) <= mse_scores[0], "No improvement in MSE"

def test_performance_benchmarking(valid_config, sample_task_data):
    """
    Performance benchmarking test to verify resource usage and scaling.
    """
    # Initialize manager
    manager = ExperimentManager(valid_config)
    manager.set_task_data(
        train_data=sample_task_data['train_data'],
        val_data=sample_task_data['val_data']
    )
    
    # Run experiment
    results = manager.run()
    
    # Verify resource usage
    assert 'resources' in results
    assert results['resources']['total_runtime'] > 0
    assert results['resources']['total_memory'] > 0
    assert results['resources']['avg_runtime'] > 0
    assert results['resources']['avg_memory'] > 0

def test_result_analysis(valid_config, sample_task_data):
    """
    Test result analysis and aggregation functionality.
    """
    # Initialize manager
    manager = ExperimentManager(valid_config)
    manager.set_task_data(
        train_data=sample_task_data['train_data'],
        val_data=sample_task_data['val_data']
    )
    
    # Run experiment
    results = manager.run()
    
    # Create aggregator and add results
    aggregator = ResultAggregator()
    for trial in results['trials']:
        aggregator.add_result(trial)
    
    # Verify analysis functions
    sensitivity = aggregator.sensitivity_analysis('metric_mse')
    assert len(sensitivity) > 0
    assert all(isinstance(v, float) for v in sensitivity.values())
    
    # Get Pareto front
    pareto_idx = aggregator.get_pareto_front(
        ['metric_mse', 'resource_runtime'],
        [True, True]
    )
    assert len(pareto_idx) > 0

def test_end_to_end_optimization(valid_config, sample_task_data):
    """
    Complete end-to-end test of the optimization process.
    """
    # Initialize manager
    manager = ExperimentManager(valid_config)
    
    # Set task data
    manager.set_task_data(
        train_data=sample_task_data['train_data'],
        val_data=sample_task_data['val_data']
    )
    
    # Run experiment
    results = manager.run()
    
    # Verify optimization results
    assert isinstance(results, dict)
    assert "trials" in results
    assert len(results["trials"]) > 0
    
    # Check for improvement in metrics
    mse_scores = [trial["metrics"]["mse"] for trial in results["trials"]]
    assert min(mse_scores) <= mse_scores[0], "No improvement in MSE" 