"""
Tests for the result aggregation and visualization module.
"""

import pytest
import numpy as np
import pandas as pd
from pyreco.reservoir_tuner.metrics.aggregator import ResultAggregator, OptimizationResult

@pytest.fixture
def sample_results():
    """Create sample optimization results for testing."""
    results = []
    for i in range(10):
        result = {
            'params': {
                'nodes': 100 + i * 10,
                'density': 0.1 + i * 0.05,
                'leakage_rate': 0.1 + i * 0.1
            },
            'metrics': {
                'mse': 0.1 - i * 0.005,
                'mae': 0.2 - i * 0.01
            },
            'resources': {
                'runtime': 10 + i,
                'memory_usage': 1000 + i * 100
            },
            'history': {
                'loss': [0.5, 0.3, 0.1]
            }
        }
        results.append(result)
    return results

@pytest.fixture
def aggregator(sample_results):
    """Create a ResultAggregator instance with sample data."""
    agg = ResultAggregator()
    for result in sample_results:
        agg.add_result(result)
    return agg

def test_add_result(aggregator):
    """Test adding results to the aggregator."""
    result = {
        'params': {'nodes': 100},
        'metrics': {'mse': 0.1},
        'resources': {'runtime': 10}
    }
    aggregator.add_result(result)
    df = aggregator._to_dataframe()
    assert len(df) == 11  # 10 sample results + 1 new result
    assert 'nodes' in df.columns
    assert 'metric_mse' in df.columns
    assert 'resource_runtime' in df.columns

def test_to_dataframe(aggregator):
    """Test conversion of results to DataFrame."""
    df = aggregator._to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert all(col in df.columns for col in ['nodes', 'density', 'leakage_rate'])
    assert all(col in df.columns for col in ['metric_mse', 'metric_mae'])
    assert all(col in df.columns for col in ['resource_runtime', 'resource_memory_usage'])

def test_pareto_front(aggregator):
    """Test Pareto front calculation."""
    objectives = ['metric_mse', 'resource_runtime']
    minimize = [True, True]
    pareto_idx = aggregator.get_pareto_front(objectives, minimize)
    
    assert isinstance(pareto_idx, list)
    assert len(pareto_idx) > 0
    assert all(isinstance(idx, int) for idx in pareto_idx)
    
    # Verify Pareto optimality
    df = aggregator._to_dataframe()
    pareto_points = df.iloc[pareto_idx]
    non_pareto_points = df.drop(df.index[pareto_idx])
    
    for _, point in pareto_points.iterrows():
        dominated = False
        for _, other in pareto_points.iterrows():
            if (all(other[objectives] <= point[objectives]) and 
                any(other[objectives] < point[objectives])):
                dominated = True
                break
        assert not dominated

def test_sensitivity_analysis(aggregator):
    """Test sensitivity analysis calculation."""
    correlations = aggregator.sensitivity_analysis('metric_mse')
    
    assert isinstance(correlations, dict)
    assert all(param in correlations for param in ['nodes', 'density', 'leakage_rate'])
    assert all(isinstance(corr, float) for corr in correlations.values())
    assert all(-1 <= corr <= 1 for corr in correlations.values())

def test_report_generation(aggregator, tmp_path):
    """Test report generation functionality."""
    report_file = tmp_path / "report.md"
    aggregator.generate_report(str(report_file))
    
    assert report_file.exists()
    content = report_file.read_text()
    
    # Check report sections
    assert "# Hyperparameter Optimization Report" in content
    assert "## Summary Statistics" in content
    assert "### Performance Metrics" in content
    assert "### Resource Usage" in content
    assert "## Best Configurations" in content
    assert "## Parameter Ranges" in content 