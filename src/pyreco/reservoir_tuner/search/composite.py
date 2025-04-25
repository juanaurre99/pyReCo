"""
Composite search strategies that combine multiple search methods.
This module implements strategies like UCB-based selection of search methods.
"""

from typing import Dict, Any, List
import numpy as np
from .strategies import SearchStrategy

class CompositeSearch(SearchStrategy):
    """
    Composite search strategy that uses UCB-based selection of multiple search methods.
    
    This strategy implements a multi-armed bandit approach where each arm corresponds
    to a different search strategy. The UCB (Upper Confidence Bound) algorithm is used
    to balance exploration and exploitation when selecting which strategy to use.
    
    Attributes:
        strategies: List of search strategies to combine
        arm_counts: Number of times each strategy has been selected
        arm_rewards: Cumulative rewards for each strategy
        arm_means: Mean reward for each strategy
        total_trials: Total number of trials conducted
        exploration_factor: Parameter controlling exploration vs exploitation
    """
    
    def __init__(self, strategies: List[SearchStrategy], exploration_factor: float = 2.0):
        """
        Initialize the composite search strategy.
        
        Args:
            strategies: List of search strategies to combine
            exploration_factor: Parameter controlling exploration vs exploitation
                              in UCB selection. Higher values encourage more exploration.
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided")
        
        self.strategies = strategies
        self.arm_counts = np.zeros(len(strategies))
        self.arm_rewards = np.zeros(len(strategies))
        self.arm_means = np.zeros(len(strategies))
        self.total_trials = 0
        self.exploration_factor = exploration_factor
        self.last_selected_arm = None
    
    def _select_arm(self) -> int:
        """
        Select an arm (strategy) using the UCB algorithm.
        
        Returns:
            int: Index of the selected strategy
        """
        if self.total_trials < len(self.strategies):
            # Initial exploration: try each strategy once
            return self.total_trials
        
        # Calculate UCB values for each arm
        ucb_values = np.zeros(len(self.strategies))
        for i in range(len(self.strategies)):
            if self.arm_counts[i] == 0:
                ucb_values[i] = float('inf')
            else:
                # UCB formula: mean + exploration_factor * sqrt(2 * ln(total_trials) / arm_count)
                exploration_term = self.exploration_factor * np.sqrt(
                    2 * np.log(self.total_trials) / self.arm_counts[i]
                )
                ucb_values[i] = self.arm_means[i] + exploration_term
        
        # Select arm with highest UCB value
        return np.argmax(ucb_values)
    
    def suggest(self) -> Dict[str, Any]:
        """
        Select a strategy using UCB and get its suggestion.
        
        Returns:
            Dict[str, Any]: Dictionary of hyperparameter names and values
        """
        # Select strategy using UCB
        self.last_selected_arm = self._select_arm()
        
        # Get suggestion from selected strategy
        return self.strategies[self.last_selected_arm].suggest()
    
    def observe(self, params: Dict[str, Any], score: float) -> None:
        """
        Update the performance tracking for the selected strategy.
        
        Args:
            params: Dictionary of hyperparameters that were tried
            score: Performance score of the trial
        """
        if self.last_selected_arm is None:
            raise RuntimeError("observe() called before suggest()")
        
        # Update arm statistics
        arm_idx = self.last_selected_arm
        self.arm_counts[arm_idx] += 1
        self.arm_rewards[arm_idx] += score
        self.arm_means[arm_idx] = self.arm_rewards[arm_idx] / self.arm_counts[arm_idx]
        self.total_trials += 1
        
        # Pass observation to selected strategy
        self.strategies[arm_idx].observe(params, score)
        
        # Reset last selected arm
        self.last_selected_arm = None
    
    def get_strategy_performance(self) -> Dict[str, Any]:
        """
        Get performance statistics for each strategy.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - counts: Number of times each strategy was selected
                - means: Mean reward for each strategy
                - total_rewards: Total reward accumulated by each strategy
        """
        return {
            "counts": self.arm_counts.tolist(),
            "means": self.arm_means.tolist(),
            "total_rewards": self.arm_rewards.tolist()
        } 