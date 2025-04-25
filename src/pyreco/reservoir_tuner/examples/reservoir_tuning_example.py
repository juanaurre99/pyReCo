"""
Example usage of the Reservoir Tuner for hyperparameter optimization.

This example demonstrates how to:
1. Create a task configuration
2. Set up the search space
3. Configure the optimization strategy
4. Run the optimization process
5. Analyze the results
"""

import numpy as np
from pathlib import Path
import tempfile
import shutil

from pyreco.reservoir_tuner.experiment.manager import ExperimentManager
from pyreco.reservoir_tuner.metrics.aggregator import ResultAggregator

def main():
    # Create output directory for results
    output_dir = Path("optimization_results")
    output_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    try:
        # Create experiment configuration
        config = {
            "experiment": {
                "name": "sine_wave_prediction",
                "description": "Predict sine wave using reservoir computing",
                "seed": 42,
                "output_dir": str(output_dir)
            },
            "task": {
                "type": "vector_to_vector",
                "name": "sine_prediction",
                "input_dim": 1,
                "output_dim": 1,
                "sequence_length": 10,
                "train_ratio": 0.8,
                "validation_ratio": 0.2
            },
            "model": {
                "type": "reservoir",
                "input_layer": {
                    "type": "dense",
                    "activation": "linear"
                },
                "reservoir_layer": {
                    "type": "reservoir",
                    "activation": "tanh"
                },
                "output_layer": {
                    "type": "dense",
                    "activation": "linear"
                }
            },
            "search_space": {
                "nodes": {"type": "int", "range": [100, 101]},
                "density": {"type": "float", "range": [0.1, 0.9]},
                "activation": {"type": "categorical", "values": ["tanh", "sigmoid"]},
                "leakage_rate": {"type": "float", "range": [0.1, 1.0]},
                "alpha": {"type": "float", "range": [0.1, 2.0]}
            },
            "optimization": {
                "strategy": "composite",
                "max_trials": 20,
                "strategies": [
                    {
                        "type": "bayesian",
                        "weight": 0.7,
                        "exploration_factor": 0.1
                    },
                    {
                        "type": "random",
                        "weight": 0.3,
                        "exploration_factor": 2.0
                    }
                ]
            },
            "metrics": {
                "primary": "mse",
                "secondary": ["mae", "rmse"],
                "resource": ["time", "memory"]
            }
        }
        
        # Initialize experiment manager
        manager = ExperimentManager(config)
        
        # Run the optimization
        print("\nStarting hyperparameter optimization...")
        results = manager.run()
        # Note: ExperimentManager automatically saves results.json and best_model.pkl
        
        # Get initial MSE from first trial
        first_trial = results['trials'][0]
        print(f"\nInitial MSE: {first_trial['metrics']['mse']:.6f}")
        
        # Analyze results
        print("\nOptimization Results:")
        print(f"Best MSE: {results['best_score']:.6f}")
        print("\nBest Configuration:")
        for param, value in results['best_config'].items():
            print(f"  {param}: {value}")
        
        # Create aggregator for detailed analysis
        aggregator = ResultAggregator()
        for trial in results['trials']:
            # Convert trial results to expected format
            formatted_trial = {
                'config': trial['hyperparameters'],
                'metrics': trial['metrics'],
                'resources': trial['resources'],
                'history': trial.get('history')
            }
            aggregator.add_result(formatted_trial)
        
        # Generate comprehensive report
        report_path = output_dir / "optimization_report.md"
        aggregator.generate_report(str(report_path))
        print(f"\nDetailed report saved to: {report_path}")
        
        # Get sensitivity analysis
        print("\nParameter Sensitivity Analysis:")
        sensitivity = aggregator.sensitivity_analysis('metric_mse')
        for param, importance in sensitivity.items():
            print(f"  {param}: {importance:.4f}")
        
        # Get Pareto front
        print("\nPareto Front Analysis:")
        pareto_idx = aggregator.get_pareto_front(
            ['metric_mse', 'resource_runtime'],
            [True, True]
        )
        print(f"Found {len(pareto_idx)} Pareto-optimal configurations")
        
        print(f"\nAll results have been saved to: {output_dir}")
        print("Files generated:")
        print("  - results.json: Raw optimization results")
        print("  - best_model.pkl: Best performing model")
        print("  - optimization_report.md: Comprehensive analysis report")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        raise
    
if __name__ == "__main__":
    main() 