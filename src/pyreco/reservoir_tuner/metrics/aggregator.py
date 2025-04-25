"""
Result aggregation and visualization module for hyperparameter optimization results.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import logging

logger = logging.getLogger(__name__)

# Optional import of seaborn for enhanced visualization
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    logger.warning("seaborn not found. Using basic matplotlib for visualization.")

@dataclass
class OptimizationResult:
    """Data class for storing optimization results."""
    params: Dict[str, Any]
    metrics: Dict[str, float]
    resources: Dict[str, float]
    history: Optional[Dict[str, Any]] = None

class ResultAggregator:
    """
    Aggregates and analyzes optimization results, providing tools for
    visualization and analysis of hyperparameter optimization experiments.
    """
    
    def __init__(self):
        """Initialize the ResultAggregator."""
        self.results: List[OptimizationResult] = []
        self._df: Optional[pd.DataFrame] = None
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """
        Add a single optimization result to the aggregator.
        
        Args:
            result: Dictionary containing trial results with keys:
                - config: Hyperparameter values
                - metrics: Performance metrics
                - resources: Resource usage metrics
                - history: Training history (optional)
        """
        opt_result = OptimizationResult(
            params=result['config'],
            metrics=result.get('metrics', {}),
            resources=result['resources'],
            history=result.get('history')
        )
        self.results.append(opt_result)
        self._df = None  # Reset cached DataFrame
    
    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis."""
        if self._df is not None:
            return self._df
            
        data = []
        for result in self.results:
            row = {
                **result.params,
                **{f"metric_{k}": v for k, v in result.metrics.items()},
                **{f"resource_{k}": v for k, v in result.resources.items()}
            }
            data.append(row)
        
        self._df = pd.DataFrame(data)
        return self._df
    
    def get_pareto_front(self, objectives: List[str], minimize: List[bool]) -> List[int]:
        """
        Identify Pareto-optimal solutions based on specified objectives.
        
        Args:
            objectives: List of column names representing objectives
            minimize: List of booleans indicating whether each objective should be minimized
            
        Returns:
            List of indices of Pareto-optimal solutions
        """
        df = self._to_dataframe()
        values = df[objectives].values
        
        # Adjust values for maximization objectives
        for i, (obj, min_flag) in enumerate(zip(objectives, minimize)):
            if not min_flag:
                values[:, i] = -values[:, i]
        
        pareto_mask = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            for j in range(len(df)):
                if i != j:
                    if np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                        pareto_mask[i] = False
                        break
        
        # Convert numpy indices to Python integers
        return [int(idx) for idx in np.where(pareto_mask)[0]]
    
    def plot_pareto_front(self, obj1: str, obj2: str, minimize: List[bool] = [True, True],
                         fig_size: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot Pareto front for two objectives.
        
        Args:
            obj1: Name of first objective
            obj2: Name of second objective
            minimize: List indicating whether each objective should be minimized
            fig_size: Figure size (width, height)
        """
        df = self._to_dataframe()
        pareto_idx = self.get_pareto_front([obj1, obj2], minimize)
        
        plt.figure(figsize=fig_size)
        if SEABORN_AVAILABLE:
            sns.scatterplot(data=df, x=obj1, y=obj2, alpha=0.5, label='All solutions')
            pareto_points = df.iloc[pareto_idx]
            sns.scatterplot(data=pareto_points, x=obj1, y=obj2, 
                          color='red', label='Pareto front')
        else:
            plt.scatter(df[obj1], df[obj2], alpha=0.5, label='All solutions')
            pareto_points = df.iloc[pareto_idx]
            plt.scatter(pareto_points[obj1], pareto_points[obj2], 
                       color='red', label='Pareto front')
        
        plt.xlabel(obj1)
        plt.ylabel(obj2)
        plt.title('Pareto Front Analysis')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def sensitivity_analysis(self, target_metric: str) -> Dict[str, float]:
        """
        Perform sensitivity analysis using Spearman correlation.
        
        Args:
            target_metric: Name of the metric to analyze sensitivity for
            
        Returns:
            Dictionary mapping parameter names to correlation coefficients
        """
        df = self._to_dataframe()
        param_cols = [col for col in df.columns 
                     if not (col.startswith('metric_') or col.startswith('resource_'))]
        
        correlations = {}
        for param in param_cols:
            corr, _ = spearmanr(df[param], df[target_metric])
            correlations[param] = corr
            
        return correlations
    
    def plot_sensitivity_analysis(self, target_metric: str, 
                                fig_size: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot sensitivity analysis results.
        
        Args:
            target_metric: Name of the metric to analyze sensitivity for
            fig_size: Figure size (width, height)
        """
        correlations = self.sensitivity_analysis(target_metric)
        
        plt.figure(figsize=fig_size)
        params = list(correlations.keys())
        coeffs = list(correlations.values())
        
        y_pos = np.arange(len(params))
        if SEABORN_AVAILABLE:
            sns.barplot(x=coeffs, y=params)
        else:
            plt.barh(y_pos, coeffs)
            plt.yticks(y_pos, params)
        
        plt.xlabel('Correlation Coefficient')
        plt.title(f'Parameter Sensitivity Analysis for {target_metric}')
        plt.grid(True)
        plt.show()
    
    def generate_report(self, output_file: str) -> None:
        """
        Generate a comprehensive report of the optimization results.
        
        Args:
            output_file: Path to save the report
        """
        df = self._to_dataframe()
        
        with open(output_file, 'w') as f:
            f.write("# Hyperparameter Optimization Report\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write("### Performance Metrics\n")
            metric_cols = [col for col in df.columns if col.startswith('metric_')]
            f.write(df[metric_cols].describe().to_markdown())
            f.write("\n\n")
            
            # Resource usage
            f.write("### Resource Usage\n")
            resource_cols = [col for col in df.columns if col.startswith('resource_')]
            f.write(df[resource_cols].describe().to_markdown())
            f.write("\n\n")
            
            # Best configurations
            f.write("## Best Configurations\n\n")
            for metric in metric_cols:
                f.write(f"### Best {metric}\n")
                best_idx = df[metric].argmin()
                best_config = df.iloc[best_idx]
                f.write(pd.DataFrame(best_config).to_markdown())
                f.write("\n\n")
            
            # Parameter ranges
            f.write("## Parameter Ranges\n\n")
            param_cols = [col for col in df.columns 
                         if not (col.startswith('metric_') or col.startswith('resource_'))]
            f.write(df[param_cols].describe().to_markdown())
            f.write("\n\n") 