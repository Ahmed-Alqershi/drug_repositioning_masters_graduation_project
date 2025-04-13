"""
Visualization utilities for training and results.
"""

import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)


def plot_training_history(histories, save_dir="results"):
    """
    Plot training and validation accuracy and loss for each fold.
    
    Args:
        histories: List of training histories from cross-validation
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Plotting training history")
    
    max_epochs = max(len(history.history['accuracy']) for history in histories)
    
    all_accuracy = np.zeros((len(histories), max_epochs))
    all_val_accuracy = np.zeros((len(histories), max_epochs))
    all_loss = np.zeros((len(histories), max_epochs))
    all_val_loss = np.zeros((len(histories), max_epochs))
    
    for i, history in enumerate(histories):
        epochs = len(history.history['accuracy'])
        all_accuracy[i, :epochs] = history.history['accuracy']
        all_val_accuracy[i, :epochs] = history.history['val_accuracy']
        all_loss[i, :epochs] = history.history['loss']
        all_val_loss[i, :epochs] = history.history['val_loss']
    
    avg_accuracy = np.mean(all_accuracy, axis=0)
    avg_val_accuracy = np.mean(all_val_accuracy, axis=0)
    avg_loss = np.mean(all_loss, axis=0)
    avg_val_loss = np.mean(all_val_loss, axis=0)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(avg_accuracy, label='Training Accuracy')
    plt.plot(avg_val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_loss, label='Training Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training history plot saved to {plot_path}")


def plot_confusion_matrix(y_true, y_pred, save_dir="results"):
    """
    Plot the confusion matrix for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Plotting confusion matrix")
    
    # Convert probabilities to binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure
    plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix plot saved to {plot_path}")


def plot_roc_curve(y_true, y_pred, save_dir="results"):
    """
    Plot the ROC curve for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Plotting ROC curve")
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save figure
    plot_path = os.path.join(save_dir, 'roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve plot saved to {plot_path}")


def plot_precision_recall_curve(y_true, y_pred, save_dir="results"):
    """
    Plot the precision-recall curve for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        save_dir: Directory to save the plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Plotting precision-recall curve")
    
    # Compute precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    
    # Save figure
    plot_path = os.path.join(save_dir, 'precision_recall_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Precision-recall curve plot saved to {plot_path}")


def plot_hyperparameter_results(results_file, save_dir="results"):
    """
    Create visualizations for hyperparameter tuning results.
    
    Args:
        results_file: Path to the CSV results file
        save_dir: Directory to save the plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Plotting hyperparameter results from {results_file}")
    
    # Read results
    df = pd.read_csv(results_file)
    
    # Extract parameter values from 'args' column if it exists
    if 'args' in df.columns:
        # Parse the JSON-like string in args column
        for idx, row in df.iterrows():
            args_dict = eval(row['args'])
            for key, value in args_dict.items():
                df.loc[idx, key] = value
    
    # Plot word embedding size vs accuracy
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='we', y='best_val_accuracy', data=df)
    plt.title('Effect of Word Vector Size on Validation Accuracy')
    plt.xlabel('Word Embedding Size')
    plt.ylabel('Best Validation Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(save_dir, 'vector_size_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot learning rate vs accuracy
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='lr', y='best_val_accuracy', data=df)
    plt.title('Effect of Learning Rate on Validation Accuracy')
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Validation Accuracy')
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(save_dir, 'learning_rate_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot CNN vs Dense model comparison
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='cnn', y='best_val_accuracy', data=df)
    plt.title('CNN vs Dense Model Comparison')
    plt.xlabel('CNN? (0=Dense, 1=CNN)')
    plt.ylabel('Best Validation Accuracy')
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(save_dir, 'model_type_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot LR Scheduler comparison
    if 'lr_scheduler' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='lr_scheduler', y='best_val_accuracy', data=df)
        plt.title('Learning Rate Scheduler Comparison')
        plt.xlabel('Learning Rate Scheduler')
        plt.ylabel('Best Validation Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(save_dir, 'lr_scheduler_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot training time by scheduler
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='lr_scheduler', y='time_taken', data=df)
        plt.title('Training Time by Learning Rate Scheduler')
        plt.xlabel('Learning Rate Scheduler')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(save_dir, 'lr_scheduler_time.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a heatmap for learning rate and scheduler
        if len(df['lr'].unique()) > 1:
            # Pivot the data for a heatmap
            pivot_df = df.pivot_table(
                index='lr', 
                columns='lr_scheduler',
                values='best_val_accuracy',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Validation Accuracy: Learning Rate vs Scheduler')
            plt.tight_layout()
            
            # Save figure
            plot_path = os.path.join(save_dir, 'lr_scheduler_heatmap.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot kernel size vs accuracy for CNN models
    if 'kernel' in df.columns:
        cnn_df = df[df['cnn'] == 1]
        if not cnn_df.empty:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='kernel', y='best_val_accuracy', data=cnn_df)
            plt.title('Effect of Kernel Size on CNN Performance')
            plt.xlabel('Kernel Size')
            plt.ylabel('Best Validation Accuracy')
            plt.tight_layout()
            
            # Save figure
            plot_path = os.path.join(save_dir, 'kernel_size_results.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Plot batch size vs accuracy
    if 'batch_size' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='batch_size', y='best_val_accuracy', data=df)
        plt.title('Effect of Batch Size on Validation Accuracy')
        plt.xlabel('Batch Size')
        plt.ylabel('Best Validation Accuracy')
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(save_dir, 'batch_size_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interaction between batch size and model type
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='batch_size', y='best_val_accuracy', hue='cnn', data=df)
        plt.title('Batch Size Effect by Model Type')
        plt.xlabel('Batch Size')
        plt.ylabel('Best Validation Accuracy')
        plt.legend(title='CNN', labels=['Dense', 'CNN'])
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(save_dir, 'batch_size_by_model.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Hyperparameter result plots saved to {save_dir}")