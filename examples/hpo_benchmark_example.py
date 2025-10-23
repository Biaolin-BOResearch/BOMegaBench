#!/usr/bin/env python3
"""
HPO Benchmarks Example

This example demonstrates how to use the HPO (Hyperparameter Optimization) benchmarks
integrated into BOMegaBench using Bayesmark's SklearnModel interface directly.

Prerequisites:
  pip install bayesmark

The HPO benchmarks provide real-world machine learning hyperparameter optimization
problems with the following features:
- Direct integration with Bayesmark's SklearnModel for standardized benchmarking
- Categorical representation of discrete hyperparameters (both integer and categorical)
- Multiple ML models (DT, RF, SVM, MLP-sgd, ada, kNN, lasso, linear)
- Multiple datasets (iris, wine, digits, breast, boston, diabetes)
- Multiple metrics (accuracy, negative log-likelihood, MSE, MAE)
- Cross-validation based evaluation with train/test split for generalization

Hyperparameter Encoding Strategy:
- Real parameters: Normalized to [0,1] range
- Integer parameters: Each possible integer value gets one dimension (argmax selection)
- Categorical parameters: Each category gets one dimension (argmax selection)

Model Names (Bayesmark convention):
- DT: Decision Tree
- RF: Random Forest  
- SVM: Support Vector Machine
- MLP-sgd: Multi-Layer Perceptron with SGD
- ada: AdaBoost
- kNN: k-Nearest Neighbors
- lasso: Lasso Regression
- linear: Linear Regression
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import numpy as np
    import bomegabench as bmb
    
    print("=== BOMegaBench HPO Benchmarks Example ===\n")
    
    # Check available suites
    suites = bmb.list_suites()
    print(f"Available suites: {suites}")
    
    if 'hpo' not in suites:
        print("\n❌ HPO benchmarks not available!")
        print("Install dependencies: pip install bayesmark scikit-learn")
        exit(1)
    
    # List HPO functions
    hpo_functions = bmb.list_functions('hpo')
    print(f"\n✓ HPO suite contains {len(hpo_functions)} functions")
    
    # Show some example functions
    print("\nExample HPO functions:")
    for i, func_name in enumerate(hpo_functions[:10]):
        print(f"  {i+1}. {func_name}")
    
    # Get a specific function (using Bayesmark naming convention)
    func_name = 'DT_iris_acc'  # DT = DecisionTree in Bayesmark
    if func_name in hpo_functions:
        print(f"\n=== Testing {func_name} ===")
        
        func = bmb.get_function(func_name, 'hpo')
        
        # Print function metadata
        print(f"Function: {func.metadata['name']}")
        print(f"Model: {func.metadata['model']}")
        print(f"Dataset: {func.metadata['dataset']}")
        print(f"Metric: {func.metadata['metric']}")
        print(f"Dimension: {func.dim}")
        print(f"Domain: {func.metadata['domain']}")
        print(f"Original hyperparameter space: {list(func.metadata['original_space'].keys())}")
        
        # Demonstrate hyperparameter encoding
        print(f"\nHyperparameter encoding details:")
        for i, dim_info in enumerate(func.continuous_space):
            if dim_info['type'] == 'cat':
                print(f"  Dim {i}: {dim_info['name']} (categorical choice: {dim_info['choice']})")
            elif dim_info['type'] == 'int_as_cat':
                print(f"  Dim {i}: {dim_info['name']} (integer choice: {dim_info['choice']})")
            else:
                print(f"  Dim {i}: {dim_info['name']} ({dim_info['type']}, bounds: {dim_info.get('original_bounds', 'N/A')})")
        
        print(f"\nEncoding strategy explanation:")
        print("- Real parameters are normalized to [0,1] and scaled back during evaluation")
        print("- Integer parameters (e.g., max_depth ∈ [1,5]) become 5 dimensions")
        print("- Categorical parameters (e.g., kernel ∈ ['linear','poly','rbf']) become 3 dimensions")
        print("- For discrete parameters, argmax selects the actual value to use")
        
        # Generate random hyperparameter configurations
        print(f"\n=== Optimization Example ===")
        print("Evaluating 5 random hyperparameter configurations...")
        
        # Generate random points in [0,1]^d
        X = torch.rand(5, func.dim)
        print(f"Random configurations (normalized to [0,1]):")
        for i, x in enumerate(X):
            print(f"  Config {i+1}: {x.numpy()}")
        
        # Evaluate the configurations
        print("\nEvaluating configurations (this may take a moment)...")
        results = func(X)
        print(f"Results (lower is better for {func.metadata['metric']}):")
        for i, (x, result) in enumerate(zip(X, results)):
            print(f"  Config {i+1}: {result.item():.4f}")
        
        # Find best configuration
        best_idx = torch.argmin(results)
        best_config = X[best_idx]
        best_result = results[best_idx]
        print(f"\nBest configuration: Config {best_idx+1}")
        print(f"Best result: {best_result.item():.4f}")
        
        # Decode the best configuration back to original hyperparameters
        print(f"\nDecoded hyperparameters for best configuration:")
        best_params = func._decode_continuous_params(best_config.numpy())
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    
    else:
        print(f"\n❌ Function {func_name} not found in HPO suite")
        print("Available functions:", hpo_functions[:5], "...")
    
    print(f"\n=== HPO Benchmarks Statistics ===")
    
    # Count functions by model type
    model_counts = {}
    dataset_counts = {}
    metric_counts = {}
    
    for func_name in hpo_functions:
        parts = func_name.split('_')
        if len(parts) >= 3:
            model = parts[0]
            dataset = parts[1] 
            metric = parts[2]
            
            model_counts[model] = model_counts.get(model, 0) + 1
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
            metric_counts[metric] = metric_counts.get(metric, 0) + 1
    
    print(f"Models: {dict(model_counts)}")
    print(f"Datasets: {dict(dataset_counts)}")
    print(f"Metrics: {dict(metric_counts)}")
    
    print(f"\n✓ HPO Benchmarks example completed successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMissing dependencies. Install with:")
    print("  pip install bayesmark scikit-learn")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 