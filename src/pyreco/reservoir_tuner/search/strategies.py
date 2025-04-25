"""
Search strategies for hyperparameter optimization.
This module contains the base SearchStrategy class and various implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import warnings

try:
    import skopt
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    warnings.warn(
        "scikit-optimize not found. BayesianSearch will fall back to RandomSearch. "
        "Install scikit-optimize for Bayesian optimization support."
    )

class SearchStrategy(ABC):
    """
    Abstract base class for hyperparameter search strategies.
    """
    
    def __init__(self, parameter_space: Dict[str, Any]):
        """
        Initialize the search strategy.
        
        Args:
            parameter_space: Dictionary defining the search space for each parameter
        """
        self.parameter_space = parameter_space
    
    @abstractmethod
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest a new set of hyperparameters to try.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameter names and values
        """
        pass
    
    @abstractmethod
    def observe(self, params: Dict[str, Any], score: float) -> None:
        """
        Observe the result of a hyperparameter trial.
        
        Args:
            params: Dictionary of hyperparameters that were tried
            score: Performance score of the trial
        """
        pass

class RandomSearch(SearchStrategy):
    """Random search strategy for hyperparameter optimization.
    
    This strategy randomly samples parameters from uniform distributions
    within specified ranges.
    
    Attributes:
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
                     specifying the range for each parameter.
    """
    
    def __init__(self, param_ranges: Dict[str, Tuple[float, float]]):
        """Initialize the random search strategy.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) tuples.
                         For example: {"learning_rate": (0.001, 0.1)}
        
        Raises:
            ValueError: If any parameter range is invalid (min > max or wrong format).
        """
        super().__init__(param_ranges)
        self.param_ranges = {}
        
        for param_name, param_range in param_ranges.items():
            if len(param_range) != 2:
                raise ValueError(f"Parameter range for {param_name} must be a tuple of (min, max)")
            
            min_val, max_val = param_range
            if min_val > max_val:
                raise ValueError(f"Minimum value {min_val} is greater than maximum value {max_val} for parameter {param_name}")
            
            self.param_ranges[param_name] = (min_val, max_val)
    
    def suggest(self) -> Dict[str, Any]:
        """Suggest a new set of hyperparameters by random sampling.
        
        Returns:
            Dict[str, Any]: A dictionary of parameter names and their randomly sampled values.
        """
        params = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            # Generate a random value in the range [min_val, max_val]
            params[param_name] = np.random.uniform(min_val, max_val)
        
        return params
    
    def observe(self, params: Dict[str, Any], score: float) -> None:
        """Observe the result of a trial.
        
        For random search, this method does nothing as the strategy
        doesn't learn from previous trials.
        
        Args:
            params: The parameters that were tried.
            score: The score achieved with these parameters.
        """
        pass  # Random search doesn't use observations

class BayesianSearch(SearchStrategy):
    """Bayesian optimization strategy for hyperparameter tuning.
    
    This strategy uses Gaussian Process-based optimization from scikit-optimize
    to suggest promising hyperparameter configurations based on previous trials.
    If scikit-optimize is not available, falls back to random search.
    
    Attributes:
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
        param_names: List of parameter names in fixed order
        optimizer: scikit-optimize Optimizer instance or None if using fallback
        fallback_strategy: RandomSearch instance used when scikit-optimize is unavailable
    """
    
    def __init__(self, param_ranges: Dict[str, Tuple[float, float]]):
        """Initialize the Bayesian optimization strategy.
        
        Args:
            param_ranges: Dictionary mapping parameter names to (min, max) tuples.
                         For example: {"learning_rate": (0.001, 0.1)}
        """
        super().__init__(param_ranges)
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.optimizer = None
        self.fallback_strategy = None
        
        if SKOPT_AVAILABLE:
            # Convert param_ranges to skopt space
            space = [
                (min_val, max_val, 'uniform') 
                for min_val, max_val in [param_ranges[name] for name in self.param_names]
            ]
            
            # Initialize the optimizer with reasonable defaults
            self.optimizer = skopt.Optimizer(
                dimensions=space,
                base_estimator="GP",  # Gaussian Process
                acq_func="EI",        # Expected Improvement
                n_initial_points=5,    # Number of random points before using GP
                random_state=None
            )
        else:
            # Fall back to random search if scikit-optimize is not available
            self.fallback_strategy = RandomSearch(param_ranges)
    
    def suggest(self) -> Dict[str, Any]:
        """Suggest a new set of hyperparameters.
        
        Returns:
            Dict[str, Any]: A dictionary of parameter names and their suggested values.
        """
        if self.optimizer is not None:
            # Get suggestion from scikit-optimize
            suggested_point = self.optimizer.ask()
            return dict(zip(self.param_names, suggested_point))
        else:
            # Fall back to random search
            return self.fallback_strategy.suggest()
    
    def observe(self, params: Dict[str, Any], score: float) -> None:
        """Observe the result of a trial.
        
        Args:
            params: The parameters that were tried.
            score: The score achieved with these parameters.
                  Higher scores are better (will be negated for minimization).
        """
        if self.optimizer is not None:
            # Convert params dict to list in correct order
            point = [params[name] for name in self.param_names]
            # scikit-optimize minimizes, so negate the score
            self.optimizer.tell(point, -score)
        else:
            # Fall back to random search (which ignores observations)
            self.fallback_strategy.observe(params, score) 