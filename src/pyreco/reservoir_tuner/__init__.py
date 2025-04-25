"""
Reservoir Computing Hyperparameter Tuning Framework.
"""

from .search.strategies import SearchStrategy
from .search.composite import CompositeSearch
from .experiment.manager import ExperimentManager
from .experiment.engine import ExecutionEngine
from .metrics.collector import MetricsCollector
from .metrics.aggregator import ResultAggregator

__version__ = '0.1.0'
__all__ = [
    'SearchStrategy',
    'CompositeSearch',
    'ExperimentManager',
    'ExecutionEngine',
    'MetricsCollector',
    'ResultAggregator'
] 