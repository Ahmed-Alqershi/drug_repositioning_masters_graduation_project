"""
Utilities for processing and aggregating results.
"""

import os
import logging
import pandas as pd
import json
import numpy as np

logger = logging.getLogger(__name__)


def save_experiment_metrics(args, histories, time_taken, save_dir="results"):
    """
    Save experiment metrics to a JSON file, appending new results.
    
    Args:
        args: Command line arguments
        histories: List of training histories from cross-validation
        time_taken: Total time taken for the experiment
        save_dir: Directory to save the metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Aggregate metrics from histories
    avg_val_accuracy = []
    avg_val_loss = []
    
    for history in histories:
        avg_val_accuracy.append(max(history.history['val_accuracy']))
        avg_val_loss.append(min(history.history['val_loss']))
    
    best_val_accuracy = round(np.mean(avg_val_accuracy) * 100, 2)
    best_val_loss = round(np.mean(avg_val_loss), 2)
    
    new_metrics = {
        "args": vars(args),
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "time_taken": time_taken
    }
    
    metrics_path = os.path.join(save_dir, "experiment_metrics.json")
    
    # Load existing metrics if the file exists
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = []
    
    # Append new metrics
    all_metrics.append(new_metrics)
    
    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    logger.info(f"Experiment metrics appended to {metrics_path}")


def aggregate_results(input_file="results/experiment_metrics.json", output_file="results/results_table.csv"):
    """
    Aggregate results from experiment_metrics.json into a single CSV file.
    
    Args:
        input_file: Path to input JSON file with experiment results
        output_file: Path to output CSV file
    """
    logger.info(f"Aggregating results from {input_file} to {output_file}")
    
    if not os.path.exists(input_file):
        logger.error(f"No results found in {input_file}")
        return
    
    # Load metrics from JSON
    with open(input_file, 'r') as f:
        all_metrics = json.load(f)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    
    logger.info(f"Results table saved to {output_file}")
    logger.info(f"Processed {len(all_metrics)} experiment runs")
    
    return results_df