"""
Random search strategy implementation for hyperparameter optimization.
"""
from typing import Dict, Any
import numpy as np
from .strategies import SearchStrategy

class RandomSearch(SearchStrategy):
    """
    Implements random search for hyperparameter optimization.
    """
    def __init__(self, parameter_space: Dict[str, Any], seed: int = None):
        """
        Initialize the random search strategy.
        
        Args:
            parameter_space: Dictionary defining the search space for each parameter
            seed: Random seed for reproducibility
        """
        super().__init__(parameter_space)
        self.rng = np.random.RandomState(seed)
        
    def suggest(self) -> Dict[str, Any]:
        """
        Suggest a random configuration from the parameter space.
        
        Returns:
            Dict containing randomly sampled parameter values
        """
        config = {}
        for param_name, param_config in self.parameter_space.items():
            param_type = param_config.get('type', 'continuous')  # Default to continuous for backward compatibility
            
            if param_type in ['continuous', 'float']:
                # Handle both min/max and range specifications
                if 'range' in param_config:
                    min_val, max_val = param_config['range']
                else:
                    min_val = param_config.get('min', 0.0)
                    max_val = param_config.get('max', 1.0)
                value = self.rng.uniform(min_val, max_val)
            elif param_type in ['discrete', 'int']:  # Handle both discrete and int types
                if 'range' in param_config:
                    min_val, max_val = param_config['range']
                else:
                    min_val = param_config.get('min', 0)
                    max_val = param_config.get('max', 10)
                value = self.rng.randint(min_val, max_val + 1)
            elif param_type == 'categorical':
                value = self.rng.choice(param_config['values'])
            else:
                raise ValueError(f"Unknown parameter type: {param_type}")
            config[param_name] = value
        return config
        
    def observe(self, params: Dict[str, Any], score: float) -> None:
        """
        Record an observation of the performance for a set of parameters.
        
        Args:
            params: Parameter configuration that was evaluated
            score: Performance score for the configuration
        """
        # Random search doesn't use past observations
        pass 