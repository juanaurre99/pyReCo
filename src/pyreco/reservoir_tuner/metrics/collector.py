"""
Metrics collection and tracking system.
This module handles performance metrics, resource usage, and reservoir properties.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from pyreco.core.network_prop_extractor import NetworkQuantifier, NodePropExtractor

class MetricsCollector:
    """
    Collects and tracks various metrics during hyperparameter optimization.
    """
    
    def __init__(self):
        """
        Initialize the metrics collector with network property extractors.
        """
        self.trials: List[Dict[str, Any]] = []
        self.network_quantifier = NetworkQuantifier()
        self.node_prop_extractor = NodePropExtractor()
    
    def add_trial(self, trial_result: Dict[str, Any], model=None) -> None:
        """
        Add a trial result to the collection and compute additional metrics.
        
        Args:
            trial_result: Dictionary containing trial results and metrics
            model: Optional RC model to extract network properties from
        """
        # Add basic trial metrics
        metrics = {
            'params': trial_result['params'],
            'metrics': trial_result['metrics'],
            'resources': trial_result['resources']
        }
        
        # Add performance metrics
        metrics['performance'] = {
            'mse': trial_result['metrics']['mse'],
            'mae': trial_result['metrics']['mae'],
            'training_time': trial_result['resources']['runtime'],
            'memory_peak': trial_result['resources']['memory_usage'],
            'cpu_usage': trial_result['resources']['cpu_percent']
        }
        
        # Add network properties if model is provided
        if model is not None and hasattr(model, 'get_reservoir_layer'):
            reservoir = model.get_reservoir_layer()
            if reservoir is not None:
                adj_matrix = reservoir.get_adjacency_matrix()
                
                # Network-level properties
                network_props = self.network_quantifier.extract_properties(adj_matrix)
                metrics['network'] = network_props
                
                # Node-level properties
                node_props = self.node_prop_extractor.extract_properties(adj_matrix)
                metrics['nodes'] = node_props
                
                # Additional reservoir metrics
                metrics['reservoir'] = {
                    'size': reservoir.nodes,
                    'activation': reservoir.activation,
                    'leakage_rate': reservoir.leakage_rate,
                    'input_scaling': reservoir.input_scaling,
                    'bias_scaling': reservoir.bias_scaling,
                    'spectral_radius': network_props['spectral_radius']
                }
        
        self.trials.append(metrics)
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert collected trials to a pandas DataFrame with flattened metrics.
        
        Returns:
            pd.DataFrame: DataFrame containing all trial results and metrics
        """
        # Flatten nested dictionaries
        flat_trials = []
        for trial in self.trials:
            flat_trial = {}
            for key, value in trial.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_trial[f"{key}_{subkey}"] = subvalue
                else:
                    flat_trial[key] = value
            flat_trials.append(flat_trial)
        
        return pd.DataFrame(flat_trials)
    
    def compute_aggregates(self) -> Dict[str, Any]:
        """
        Compute comprehensive aggregate statistics from the collected trials.
        
        Returns:
            Dict[str, Any]: Dictionary of aggregate statistics including:
                - Performance metrics (MSE, MAE)
                - Resource usage (runtime, memory, CPU)
                - Network properties
                - Best configurations
        """
        df = self.get_dataframe()
        
        aggregates = {
            'performance': {
                'mean_mse': df['metrics_mse'].mean(),
                'std_mse': df['metrics_mse'].std(),
                'mean_mae': df['metrics_mae'].mean(),
                'std_mae': df['metrics_mae'].std(),
                'mean_runtime': df['resources_runtime'].mean(),
                'mean_memory': df['resources_memory_usage'].mean(),
                'mean_cpu': df['resources_cpu_percent'].mean()
            },
            'best_trial': {
                'params': df.loc[df['metrics_mse'].idxmin(), 'params'],
                'mse': df['metrics_mse'].min(),
                'mae': df.loc[df['metrics_mse'].idxmin(), 'metrics_mae']
            }
        }
        
        # Add network property statistics if available
        network_cols = [col for col in df.columns if col.startswith('network_')]
        if network_cols:
            aggregates['network'] = {
                col.replace('network_', ''): {
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
                for col in network_cols
            }
        
        return aggregates
    
    def get_pareto_front(self) -> pd.DataFrame:
        """
        Compute the Pareto front of non-dominated solutions based on MSE and resource usage.
        
        Returns:
            pd.DataFrame: DataFrame containing Pareto-optimal solutions
        """
        df = self.get_dataframe()
        
        # Consider MSE and resource usage for Pareto optimality
        objectives = ['metrics_mse', 'resources_memory_usage', 'resources_runtime']
        
        # Initialize all points as non-dominated
        is_pareto = np.ones(len(df), dtype=bool)
        
        # Compare each point with every other point
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:
                    # Check if point j dominates point i
                    dominates = True
                    for obj in objectives:
                        if df.iloc[j][obj] > df.iloc[i][obj]:
                            dominates = False
                            break
                    if dominates and any(df.iloc[j][obj] < df.iloc[i][obj] for obj in objectives):
                        is_pareto[i] = False
                        break
        
        return df[is_pareto]
    
    def get_sensitivity_analysis(self) -> Dict[str, float]:
        """
        Perform sensitivity analysis to determine parameter importance.
        
        Returns:
            Dict[str, float]: Dictionary mapping parameters to their importance scores
        """
        df = self.get_dataframe()
        params = eval(df['params'].iloc[0])  # Get parameter names from first trial
        
        sensitivity = {}
        for param in params.keys():
            # Extract parameter values and corresponding MSE
            param_values = df['params'].apply(lambda x: eval(x)[param])
            mse_values = df['metrics_mse']
            
            # Calculate correlation coefficient
            correlation = np.corrcoef(param_values, mse_values)[0, 1]
            sensitivity[param] = abs(correlation)
        
        return sensitivity 