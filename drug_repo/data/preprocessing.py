"""
Data preprocessing functions for the drug repositioning project.
"""

import os
import logging
from itertools import product
import pandas as pd

logger = logging.getLogger(__name__)

def extract_positive_data(df):
    """
    Extract positive drug-disease pairs from the dataset.
    
    Args:
        df: DataFrame containing the raw data
        
    Returns:
        DataFrame containing filtered positive pairs
    """
    logger.info("Extracting positive drug-disease pairs")
    pos_df = df[
        (df.SUBJECT_SEMTYPE.str.contains("phsu"))
        & (df.OBJECT_SEMTYPE.str.contains("dsyn"))
    ]
    pos_df = pos_df.drop(["frequency", "Source"], axis=1)
    pos_df = pos_df.reset_index(drop=True)
    
    logger.info(f"Extracted {len(pos_df)} positive drug-disease pairs")
    return pos_df


def create_drug_and_disease_list(df):
    """
    Create lists of unique drugs and diseases from the dataset.
    
    Args:
        df: DataFrame containing drug-disease pairs
        
    Returns:
        tuple: (drug_list, disease_list)
    """
    drug_list = df.SUBJECT_CUI.unique()
    disease_list = df.OBJECT_CUI.unique()
    
    logger.info(f"Number of unique drugs: {len(drug_list)}")
    logger.info(f"Number of unique diseases: {len(disease_list)}")

    return drug_list, disease_list


def create_negative_dataset(drugs, diseases, positive_data, save=True, balanced=True):
    """
    Create a negative dataset of drug-disease pairs.
    
    Args:
        drugs: List of drug identifiers
        diseases: List of disease identifiers
        positive_data: DataFrame containing positive pairs
        balanced: Boolean indicating if the negative dataset should be balanced
        
    Returns:
        DataFrame containing negative drug-disease pairs
    """
    # Check if dataset already exists
    if balanced:
        neg_path = "drug_repo/data/balanced_negative_dataset.csv"
    else:
        neg_path = "drug_repo/data/imbalanced_negative_dataset.csv"

    if os.path.exists(neg_path):
        logger.info(f"Loading existing negative dataset from {neg_path}")
        return pd.read_csv(neg_path)
    
    logger.info("Generating negative dataset")
    
    # Generate All Drug-Disease Pairs
    all_pairs = list(product(drugs, diseases))
    all_pairs_df = pd.DataFrame(all_pairs, columns=["SUBJECT_CUI", "OBJECT_CUI"])

    # Remove Existing Positive Pairs
    positive_pairs = set(zip(positive_data["SUBJECT_CUI"], positive_data["OBJECT_CUI"]))
    all_pairs_df["Pair"] = list(
        zip(all_pairs_df["SUBJECT_CUI"], all_pairs_df["OBJECT_CUI"])
    )
    negative_pairs_df = all_pairs_df[~all_pairs_df["Pair"].isin(positive_pairs)]

    # Balance Negative Dataset
    num_positive = len(positive_pairs)

    if balanced:
        negative_pairs = negative_pairs_df.sample(
            n=min(num_positive, len(negative_pairs_df))
        )

    else:
        negative_pairs = negative_pairs_df.sample(
            n=min(num_positive*2, len(negative_pairs_df))
        )

        negative_pairs = negative_pairs.drop(
            columns=["Pair"]
        ).reset_index(drop=True)

        # Save the Dataset
        if save:
            negative_pairs.to_csv(neg_path, index=False)
            logger.info(f"Saved {len(negative_pairs)} negative pairs to {neg_path}")

        return negative_pairs


def create_other_relations(df):
    """
    Create a dataset of relations that are not drug-disease pairs.
    
    Args:
        df: DataFrame containing the raw data
        
    Returns:
        DataFrame containing other relations
    """
    logger.info("Extracting other relation types")
    other_relations = df[
        ~(df.SUBJECT_SEMTYPE.str.contains("phsu"))
        & ~(df.OBJECT_SEMTYPE.str.contains("dsyn"))
    ]
    other_relations = other_relations.drop(["frequency", "Source"], axis=1)
    other_relations = other_relations.reset_index(drop=True)
    
    logger.info(f"Extracted {len(other_relations)} other relation types")
    return other_relations