"""
Word embedding functions for the drug repositioning project.
"""

import multiprocessing
import logging
import numpy as np
import pandas as pd
from gensim.models import FastText
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _create_pos_strs(positive_data):
    """
    Create positive strings for the Word Embedding model.
    
    Args:
        positive_data: DataFrame containing positive drug-disease pairs
        
    Returns:
        tuple: (predicate_strings, subject_semtype_strings, object_semtype_strings)
    """
    pos_preds = positive_data.iloc[:, 0:3]
    pos_sub_semtype = positive_data.iloc[:, [0, 3]]
    pos_obj_semtype = positive_data.iloc[:, [2, 4]]

    pos_pred_str = []
    for i in range(len(pos_preds)):
        temp = list(pos_preds.iloc[i, :].astype(str))
        temp = " ".join(temp)
        pos_pred_str.append(temp)

    pos_s_semtype_str = []
    added_string = "has_semtype"
    for row in pos_sub_semtype.itertuples(index=False):
        sub = row[0]
        obj = row[1].replace(",", " and ")
        pos_s_semtype_str.append(f"{sub} {added_string} {obj}")

    pos_o_semtype_str = []
    for row in pos_obj_semtype.itertuples(index=False):
        sub = row[0]
        obj = row[1].replace(",", " and ")
        pos_o_semtype_str.append(f"{sub} {added_string} {obj}")

    return pos_pred_str, pos_s_semtype_str, pos_o_semtype_str


def _create_neg_strs(balanced_negative_pairs):
    """
    Create negative strings for the Word Embedding model.
    
    Args:
        balanced_negative_pairs: DataFrame containing negative drug-disease pairs
        
    Returns:
        list: Negative predicate strings
    """
    neg_pred_str = []
    added_string = "neg_treat"
    for row in balanced_negative_pairs.itertuples(index=False):
        sub = row[0]
        obj = row[1].replace(",", " and ")
        neg_pred_str.append(f"{sub} {added_string} {obj}")

    return neg_pred_str


def _create_other_strs(other_relations):
    """
    Create other relation strings for the Word Embedding model.
    
    Args:
        other_relations: DataFrame containing other relations
        
    Returns:
        tuple: (predicate_strings, subject_semtype_strings, object_semtype_strings)
    """
    other_preds = other_relations.iloc[:, 0:3]
    other_sub_semtype = other_relations.iloc[:, [0, 3]]
    other_obj_semtype = other_relations.iloc[:, [2, 4]]

    other_pred_str = []
    for i in range(len(other_preds)):
        temp = list(other_preds.iloc[i, :].astype(str))
        temp = " ".join(temp)
        other_pred_str.append(temp)

    other_s_semtype_str = []
    added_string = "has_semtype"
    for row in other_sub_semtype.itertuples(index=False):
        sub = row[0]
        obj = row[1].replace(",", " and ")
        other_s_semtype_str.append(f"{sub} {added_string} {obj}")

    other_o_semtype_str = []
    for row in other_obj_semtype.itertuples(index=False):
        sub = row[0]
        obj = row[1].replace(",", " and ")
        other_o_semtype_str.append(f"{sub} {added_string} {obj}")

    return other_pred_str, other_s_semtype_str, other_o_semtype_str


def create_str_for_we(positive_data, balanced_negative_pairs, other_relations):
    """
    Create strings for the Word Embedding model.
    
    Args:
        positive_data: DataFrame containing positive drug-disease pairs
        balanced_negative_pairs: DataFrame containing negative drug-disease pairs
        other_relations: DataFrame containing other relations
        
    Returns:
        list: Combined strings for word embedding
    """
    logger.info("Creating strings for word embedding")
    
    pos_pred_str, pos_s_semtype_str, pos_o_semtype_str = _create_pos_strs(positive_data)
    neg_pred_str = _create_neg_strs(balanced_negative_pairs)
    other_pred_str, other_s_semtype_str, other_o_semtype_str = _create_other_strs(
        other_relations
    )

    all_strings = (
        pos_pred_str
        + pos_s_semtype_str
        + pos_o_semtype_str
        + neg_pred_str
        + other_pred_str
        + other_s_semtype_str
        + other_o_semtype_str
    )
    
    logger.info(f"Created {len(all_strings)} strings for word embedding")
    return all_strings


def tokenize_strings(combined_list_of_strings):
    """
    Tokenize/preprocess the strings into words.
    
    Args:
        combined_list_of_strings: List of strings to tokenize
        
    Returns:
        list: Tokenized strings
    """
    logger.info("Tokenizing strings for word embedding")
    
    tokenized_strings = []
    for s in combined_list_of_strings:
        for i in sent_tokenize(s):
            temp = []
            # tokenize the sentence into words
            for j in word_tokenize(i):
                temp.append(j.lower())
            tokenized_strings.append(temp)
    
    logger.info(f"Tokenized into {len(tokenized_strings)} sentences")
    return tokenized_strings


def build_fast_text_model(tokenized_strings, inp_shape, window=2, min_count=1, sg=1):
    """
    Build the FastText model.
    
    Args:
        tokenized_strings: List of tokenized strings
        inp_shape: Size of the word vectors
        window: Maximum distance between the current and predicted word
        min_count: Ignores all words with total frequency lower than this
        sg: 1 for skip-gram; 0 for CBOW model
        
    Returns:
        FastText model
    """
    logger.info(f"Building FastText model (vector_size={inp_shape}, window={window}, min_count={min_count}, sg={sg})")
    
    model = FastText(
        tokenized_strings,
        vector_size=inp_shape,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count(),
        sg=sg,
    )
    
    logger.info("FastText model built successfully")
    return model


def post_we_processing(we_model, pos_preds, balanced_negative_pairs):
    """
    Perform post-processing for Word Embeddings.
    
    Args:
        we_model: FastText model
        pos_preds: DataFrame containing positive drug-disease pairs
        balanced_negative_pairs: DataFrame containing negative drug-disease pairs
        
    Returns:
        tuple: (drugs_array, disease_array, labels_array)
    """
    logger.info("Processing word embeddings for training")
    
    def get_word_vec(word):
        return we_model.wv.get_vector(word)

    pos_drugs_vec = pos_preds.SUBJECT_CUI.apply(get_word_vec)
    pos_disease_vec = pos_preds.OBJECT_CUI.apply(get_word_vec)
    pos_label = [1 for _ in range(pos_drugs_vec.shape[0])]

    neg_drugs_vec = balanced_negative_pairs.SUBJECT_CUI.apply(get_word_vec)
    neg_disease_vec = balanced_negative_pairs.OBJECT_CUI.apply(get_word_vec)
    neg_label = [0 for _ in range(neg_drugs_vec.shape[0])]

    drugs_vec = pd.concat([pos_drugs_vec, neg_drugs_vec], axis=0)
    disease_vec = pd.concat([pos_disease_vec, neg_disease_vec], axis=0)
    labels = pos_label + neg_label

    drugs_arr = np.array(drugs_vec.tolist())
    disease_arr = np.array(disease_vec.tolist())
    labels_arr = np.array(labels)
    
    logger.info(f"Created training arrays with {len(labels_arr)} samples")
    return drugs_arr, disease_arr, labels_arr


def split_data_for_training(drugs_arr, disease_arr, labels_arr, test_size=0.1):
    """
    Split the data into training and validation sets.
    
    Args:
        drugs_arr: Array of drug vectors
        disease_arr: Array of disease vectors
        labels_arr: Array of labels
        test_size: Proportion of the dataset to include in the validation split
        
    Returns:
        tuple: Training and validation data
    """
    logger.info(f"Splitting data for training (test_size={test_size})")
    
    # Combine the arrays into a single array for shuffling
    combined = list(zip(drugs_arr, disease_arr, labels_arr))

    # Split the combined array into training and testing sets
    train, validate = train_test_split(combined, test_size=test_size, shuffle=True)

    # Unzip the training and testing sets
    drugs_train, disease_train, labels_train = zip(*train)
    drugs_val, disease_val, labels_val = zip(*validate)

    # Convert back to numpy arrays
    drugs_train = np.array(drugs_train)
    disease_train = np.array(disease_train)
    labels_train = np.array(labels_train)
    drugs_val = np.array(drugs_val)
    disease_val = np.array(disease_val)
    labels_val = np.array(labels_val)
    
    logger.info(f"Training set: {len(labels_train)} samples, Validation set: {len(labels_val)} samples")
    return drugs_train, disease_train, labels_train, drugs_val, disease_val, labels_val