"""
Main entry point for the Drug Repositioning project.
"""

import argparse
import logging
import os
import pickle
import time
import pandas as pd

from drug_repo.data.preprocessing import (
    extract_positive_data,
    create_drug_and_disease_list,
    create_negative_dataset,
    create_other_relations
)
from drug_repo.utils.embeddings import (
    create_str_for_we,
    tokenize_strings,
    build_fast_text_model,
    post_we_processing,
    split_data_for_training
)
from drug_repo.models.siamese import build_snn, model_compilation_training
from drug_repo.utils.results import save_experiment_metrics
from drug_repo.visualization.plots import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)


def setup_logging(log_level=logging.INFO):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler("logs/drug_repo.log"),
            logging.StreamHandler()
        ]
    )


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Drug-Repositioning project")
    parser.add_argument('--neg_balanced', type=int, default=1, help='1 for balanced negative dataset, 0 for unbalanced')
    parser.add_argument('--we', type=int, default=1024, help='Word Embedding vector size')
    parser.add_argument('--win', type=int, default=2, help='Window size for Word Embedding')
    parser.add_argument('--min_cnt', type=int, default=1, help='Words minimum count for Word Embedding')
    parser.add_argument('--sg', type=int, default=1, help='1; Skip-gram model, 0; CBOW model (Word Embedding)')
    parser.add_argument('--depth', type=int, default=64, help='Depth of each layer of the Siamese Neural Network')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate for training')
    parser.add_argument('--lr_scheduler', type=str, default='reduce_on_plateau', 
                        choices=['constant', 'reduce_on_plateau', 'exponential', 'step', 'cosine', 'cyclic', 'one_cycle'],
                        help='Learning rate scheduler type')
    parser.add_argument('--cnn', type=int, default=0, help='1; Use CNN, 0; Use Sequential model')
    parser.add_argument('--kernel', type=int, default=3, help='Kernel size for CNN')
    parser.add_argument('--test', action='store_true', help='Evaluate model on test set')
    
    return parser.parse_args()


def main():
    """
    Main function to run the drug repositioning pipeline.
    """
    # Set up logging and parse arguments
    setup_logging()
    args = parse_arguments()
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Drug Repositioning Pipeline")
    
    start = time.time()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    
    #############################################
    # Data Preprocessing
    #############################################
    logger.info("Starting data preprocessing")
    
    # Load the Data
    data_dir = "drug_repo/data"
    pos_df = pd.read_csv(f"{data_dir}/positive_dataset.csv")
    test_df = pd.read_csv(f"{data_dir}/test_dataset.csv")
    
    # Extract Positive Data
    positive_data = extract_positive_data(pos_df)
    
    # Create drug and disease lists
    drugs, diseases = create_drug_and_disease_list(positive_data)
    
    # Create Negative Dataset
    negative_path = f"{data_dir}/negative_dataset.csv"
    balanced_negative_pairs = create_negative_dataset(
        drugs, diseases, positive_data, save=True, balanced=args.neg_balanced
    )
    
    # Create Other Relations dataset
    other_relations = create_other_relations(pos_df)
    
    #############################################
    # Word Embedding
    #############################################
    logger.info("Starting word embedding process")
    
    # Create Strings from Data for Word Embedding
    list_of_str = create_str_for_we(
        positive_data, balanced_negative_pairs, other_relations
    )
    
    # Tokenize the Strings
    tokenized_str = tokenize_strings(list_of_str)
    
    # Build FastText model
    ft_model = build_fast_text_model(
        tokenized_str, args.we, window=args.win, min_count=args.min_cnt, sg=args.sg
    )
    
    # Save FastText model for later use in predictions
    logger.info("Saving FastText model for later use")
    fasttext_model_path = "results/models/fasttext_model.pkl"
    with open(fasttext_model_path, 'wb') as f:
        pickle.dump(ft_model, f)
    logger.info(f"FastText model saved to {fasttext_model_path}")
    
    #############################################
    # Prepare Word Embeddings for training
    #############################################
    logger.info("Preparing data for training")
    
    # Post-process the FastText model results
    drugs_arr, disease_arr, labels_arr = post_we_processing(
        ft_model, positive_data, balanced_negative_pairs
    )
    
    # Split the data into training and testing sets
    drugs_train, disease_train, labels_train, drugs_val, disease_val, labels_val = (
        split_data_for_training(drugs_arr, disease_arr, labels_arr)
    )
    
    #############################################
    # Build and Train Siamese Neural Network
    #############################################
    logger.info("Building and training Siamese Neural Network")
    
    # Build the Siamese Neural Network model
    snn_model = build_snn(depth=args.depth, inp_shape=args.we, cnn=args.cnn, kernel=args.kernel)
    
    # Compile and train the model
    history = model_compilation_training(
        snn_model, drugs_train, disease_train, labels_train,
        lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, cnn=args.cnn,
        model_dir="results/models", lr_scheduler_type=args.lr_scheduler, n_splits=5
    )
    
    #############################################
    # Visualizations and Results
    #############################################
    logger.info("Creating visualizations")
    
    # Plot training history
    plot_training_history(history)
    
    # Get predictions for validation set
    if args.cnn:
        drugs_val = drugs_val.reshape(drugs_val.shape[0], drugs_val.shape[1], 1)
        disease_val = disease_val.reshape(disease_val.shape[0], disease_val.shape[1], 1)
    
    val_predictions = snn_model.predict([drugs_val, disease_val])
    
    # Create evaluation plots
    plot_confusion_matrix(labels_val, val_predictions)
    plot_roc_curve(labels_val, val_predictions)
    plot_precision_recall_curve(labels_val, val_predictions)
    
    # Record execution time
    end = time.time()
    time_taken = round(end - start, 2)
    
    # Save metrics to file
    save_experiment_metrics(args, history, time_taken)
    
    logger.info(f"Pipeline completed in {time_taken} seconds")


if __name__ == "__main__":
    main()