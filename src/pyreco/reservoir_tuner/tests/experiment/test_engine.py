"""
Test cases for the ExecutionEngine class.
"""

import pytest
import numpy as np
from pyreco.reservoir_tuner.experiment.engine import ExecutionEngine

@pytest.fixture
def sample_task_config():
    """Create a sample task configuration for testing."""
    # Generate sample data
    np.random.seed(42)
    X_train = np.random.randn(10, 20, 3)  # (n_batch, n_time, n_features)
    y_train = np.random.randn(10, 20, 2)  # (n_batch, n_time, n_features_out)
    X_val = np.random.randn(5, 20, 3)
    y_val = np.random.randn(5, 20, 2)
    
    return {
        'input_shape': (20, 3),
        'output_shape': (20, 2),
        'train_data': (X_train, y_train),
        'val_data': (X_val, y_val)
    }

@pytest.fixture
def sample_params():
    """Create sample hyperparameters for testing."""
    return {
        'nodes': 100,
        'density': 0.1,
        'activation': 'tanh',
        'leakage_rate': 0.1,
        'fraction_input': 0.5,
        'fraction_out': 0.99,
        'alpha': 0.1
    }

def test_engine_initialization(sample_task_config):
    """Test that the engine initializes correctly with task config."""
    engine = ExecutionEngine(sample_task_config)
    assert engine.input_shape == (20, 3)
    assert engine.output_shape == (20, 2)
    assert engine.X_train.shape == (10, 20, 3)
    assert engine.y_train.shape == (10, 20, 2)
    assert engine.X_val.shape == (5, 20, 3)
    assert engine.y_val.shape == (5, 20, 2)

def test_model_creation(sample_task_config, sample_params):
    """Test that model creation works with valid parameters."""
    engine = ExecutionEngine(sample_task_config)
    model = engine._create_model(sample_params)
    
    # Verify model structure
    assert model.input_layer is not None
    assert model.reservoir_layer is not None
    assert model.readout_layer is not None
    
    # Verify parameter settings
    assert model.reservoir_layer.nodes == 100
    assert model.reservoir_layer.density == 0.1
    assert model.reservoir_layer.activation == 'tanh'
    assert model.reservoir_layer.leakage_rate == 0.1
    assert model.reservoir_layer.fraction_input == 0.5
    assert model.readout_layer.fraction_out == 0.99

def test_model_training(sample_task_config, sample_params):
    """Test that model training works and returns history."""
    engine = ExecutionEngine(sample_task_config)
    model = engine._create_model(sample_params)
    history = engine._train_model(model, n_init=2)
    
    # Verify training history structure
    assert 'init_res_states' in history
    assert 'readout_weights' in history
    assert 'train_scores' in history
    assert len(history['train_scores']) == 2  # Two initializations

def test_model_evaluation(sample_task_config, sample_params):
    """Test that model evaluation works and returns metrics."""
    engine = ExecutionEngine(sample_task_config)
    model = engine._create_model(sample_params)
    engine._train_model(model)
    metrics = engine._evaluate_model(model)
    
    # Verify metrics structure
    assert 'mse' in metrics
    assert 'mae' in metrics
    assert isinstance(metrics['mse'], float)
    assert isinstance(metrics['mae'], float)

def test_trial_execution(sample_task_config, sample_params):
    """Test complete trial execution including resource monitoring."""
    engine = ExecutionEngine(sample_task_config)
    results = engine.run_trial(sample_params)
    
    # Verify results structure
    assert 'params' in results
    assert 'metrics' in results
    assert 'history' in results
    assert 'resources' in results
    
    # Verify resource monitoring
    resources = results['resources']
    assert 'runtime' in resources
    assert 'memory_usage' in resources
    assert 'cpu_percent' in resources
    assert resources['runtime'] > 0
    assert resources['memory_usage'] > 0
    assert resources['cpu_percent'] >= 0

def test_error_handling(sample_task_config):
    """Test that the engine handles errors gracefully."""
    # Create invalid parameters
    invalid_params = {
        'nodes': -100,  # Invalid number of nodes
        'density': 2.0,  # Invalid density
        'activation': 'invalid',  # Invalid activation
        'leakage_rate': 2.0,  # Invalid leakage rate
        'fraction_input': 2.0,  # Invalid fraction
        'fraction_out': 2.0  # Invalid fraction
    }
    
    engine = ExecutionEngine(sample_task_config)
    results = engine.run_trial(invalid_params)
    
    # Verify error handling
    assert 'error' in results
    assert 'resources' in results
    assert results['resources']['runtime'] > 0
    assert results['resources']['memory_usage'] > 0 