"""
Tests for the metrics collection and tracking system.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from pyreco.reservoir_tuner.metrics.collector import MetricsCollector
from pyreco.core.custom_models import RC
from pyreco.core.layers import RandomReservoirLayer

@pytest.fixture
def sample_trial_result():
    """Create a sample trial result for testing."""
    return {
        'params': {
            'nodes': 100,
            'density': 0.1,
            'activation': 'tanh',
            'leakage_rate': 0.3,
            'fraction_input': 0.2,
            'fraction_out': 0.8
        },
        'metrics': {
            'mse': 0.05,
            'mae': 0.02
        },
        'resources': {
            'runtime': 1.5,
            'memory_usage': 1024,
            'cpu_percent': 50.0
        }
    }

@pytest.fixture
def mock_model():
    """Create a mock RC model with a reservoir layer."""
    model = MagicMock(spec=RC)
    reservoir = MagicMock(spec=RandomReservoirLayer)
    
    # Set reservoir properties
    reservoir.nodes = 100
    reservoir.activation = 'tanh'
    reservoir.leakage_rate = 0.3
    reservoir.input_scaling = 1.0
    reservoir.bias_scaling = 1.0
    
    # Mock adjacency matrix
    adj_matrix = np.random.rand(100, 100)
    adj_matrix[adj_matrix < 0.9] = 0  # Create sparse matrix
    reservoir.get_adjacency_matrix.return_value = adj_matrix
    
    model.get_reservoir_layer.return_value = reservoir
    return model

def test_metrics_collector_initialization():
    """Test MetricsCollector initialization."""
    collector = MetricsCollector()
    assert len(collector.trials) == 0
    assert hasattr(collector, 'network_quantifier')
    assert hasattr(collector, 'node_prop_extractor')

def test_add_trial_without_model(sample_trial_result):
    """Test adding a trial without model information."""
    collector = MetricsCollector()
    collector.add_trial(sample_trial_result)
    
    assert len(collector.trials) == 1
    trial = collector.trials[0]
    
    assert 'params' in trial
    assert 'metrics' in trial
    assert 'resources' in trial
    assert 'performance' in trial
    
    # Check performance metrics
    assert trial['performance']['mse'] == 0.05
    assert trial['performance']['mae'] == 0.02
    assert trial['performance']['training_time'] == 1.5
    assert trial['performance']['memory_peak'] == 1024
    assert trial['performance']['cpu_usage'] == 50.0

def test_add_trial_with_model(sample_trial_result, mock_model):
    """Test adding a trial with model information."""
    collector = MetricsCollector()
    collector.add_trial(sample_trial_result, mock_model)
    
    assert len(collector.trials) == 1
    trial = collector.trials[0]
    
    # Check all metric categories are present
    assert 'network' in trial
    assert 'nodes' in trial
    assert 'reservoir' in trial
    
    # Check reservoir properties
    assert trial['reservoir']['size'] == 100
    assert trial['reservoir']['activation'] == 'tanh'
    assert trial['reservoir']['leakage_rate'] == 0.3

def test_get_dataframe(sample_trial_result):
    """Test converting trials to DataFrame."""
    collector = MetricsCollector()
    collector.add_trial(sample_trial_result)
    
    df = collector.get_dataframe()
    assert not df.empty
    assert 'metrics_mse' in df.columns
    assert 'resources_runtime' in df.columns
    assert 'performance_cpu_usage' in df.columns

def test_compute_aggregates(sample_trial_result):
    """Test computing aggregate statistics."""
    collector = MetricsCollector()
    collector.add_trial(sample_trial_result)
    
    aggregates = collector.compute_aggregates()
    assert 'performance' in aggregates
    assert 'best_trial' in aggregates
    
    # Check performance metrics
    performance = aggregates['performance']
    assert 'mean_mse' in performance
    assert 'mean_mae' in performance
    assert 'mean_runtime' in performance
    
    # Check best trial
    best_trial = aggregates['best_trial']
    assert 'params' in best_trial
    assert 'mse' in best_trial
    assert 'mae' in best_trial

def test_get_pareto_front(sample_trial_result):
    """Test computing Pareto front."""
    collector = MetricsCollector()
    
    # Add multiple trials with different trade-offs
    trial1 = dict(sample_trial_result)
    trial1['metrics']['mse'] = 0.05
    trial1['resources']['runtime'] = 1.0
    trial1['resources']['memory_usage'] = 1000
    
    trial2 = dict(sample_trial_result)
    trial2['metrics']['mse'] = 0.03
    trial2['resources']['runtime'] = 1.5
    trial2['resources']['memory_usage'] = 1200
    
    trial3 = dict(sample_trial_result)
    trial3['metrics']['mse'] = 0.07
    trial3['resources']['runtime'] = 0.8
    trial3['resources']['memory_usage'] = 800
    
    collector.add_trial(trial1)
    collector.add_trial(trial2)
    collector.add_trial(trial3)
    
    pareto_front = collector.get_pareto_front()
    assert not pareto_front.empty
    assert len(pareto_front) <= 3  # Some solutions might dominate others

def test_get_sensitivity_analysis(sample_trial_result):
    """Test sensitivity analysis."""
    collector = MetricsCollector()
    
    # Add multiple trials with different parameters
    for _ in range(5):
        trial = dict(sample_trial_result)
        trial['params'] = {
            'nodes': np.random.randint(50, 150),
            'density': np.random.uniform(0.1, 0.5),
            'leakage_rate': np.random.uniform(0.1, 0.9)
        }
        trial['metrics']['mse'] = np.random.uniform(0.01, 0.1)
        collector.add_trial(trial)
    
    sensitivity = collector.get_sensitivity_analysis()
    assert len(sensitivity) == 3  # One score per parameter
    assert all(0 <= score <= 1 for score in sensitivity.values()) 