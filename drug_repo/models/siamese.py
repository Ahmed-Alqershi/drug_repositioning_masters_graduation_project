"""
Siamese Neural Network model for drug repositioning.
"""

import logging
import os
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras import models, activations, optimizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D, Concatenate
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    LearningRateScheduler, CSVLogger
)
from tensorflow.keras.optimizers.schedules import (
    ExponentialDecay, PiecewiseConstantDecay, CosineDecay
)
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.utils import class_weight

logger = logging.getLogger(__name__)


def _sequential_block(depth):
    """
    Create a sequential block with dense layer, batch normalization, and dropout.
    
    Args:
        depth: Number of neurons in the dense layer
        
    Returns:
        Sequential model block
    """
    return models.Sequential(
        [
            Dense(depth, activation=activations.relu),
            BatchNormalization(),
            Dropout(0.1),
        ]
    )


def _cnn_block(filters, kernel_size):
    """
    Create a CNN block with convolution, batch normalization, max pooling, and dropout.
    
    Args:
        filters: Number of filters in the convolution
        kernel_size: Size of the convolution kernel
        
    Returns:
        Sequential model block
    """
    return models.Sequential(
        [
            Conv1D(int(filters), kernel_size, activation=activations.relu, padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.1),
        ]
    )


def build_snn(depth=64, inp_shape=1024, cnn=False, kernel=3):
    """
    Build the Siamese Neural Network model.
    
    Args:
        depth: Base depth of the network layers
        inp_shape: Input shape of the word vectors
        cnn: Whether to use CNN architecture (True) or dense layers (False)
        kernel: Kernel size for CNN layers
        
    Returns:
        Compiled SNN model
    """
    model_type = "CNN" if cnn else "Dense"
    logger.info(f"Building {model_type} Siamese Neural Network (depth={depth}, input_shape={inp_shape})")

    if cnn:
        drug_input = tf.keras.Input(shape=(inp_shape, 1), name="drug")
        disease_input = tf.keras.Input(shape=(inp_shape, 1), name="disease")
        features_seq = models.Sequential(
            [
                _cnn_block(depth / 2, kernel),
                _cnn_block(depth, kernel),
                _cnn_block(depth * 2, kernel),
                _cnn_block(depth * 4, kernel),
                _cnn_block(depth * 8, kernel),
                GlobalMaxPooling1D(),
            ],
            name="features",
        )
    else:
        drug_input = tf.keras.Input(shape=(inp_shape,), name="drug")
        disease_input = tf.keras.Input(shape=(inp_shape,), name="disease")
        features_seq = models.Sequential(
            [
                _sequential_block(depth),
                _sequential_block(depth * 2),
                _sequential_block(depth * 4),
                _sequential_block(depth * 8),
            ],
            name="features",
        )

    drug_features = features_seq(drug_input)
    disease_features = features_seq(disease_input)

    concatenated = Concatenate(name="concat_features")(
        [drug_features, disease_features]
    )

    dense = Dense(depth * 2, activation=activations.relu, name="learn_non_linearities")(
        concatenated
    )
    dense2 = Dense(depth, activation=activations.relu, name="learn_non_linearities2")(
        dense
    )

    output = Dense(1, activation=activations.sigmoid, name="output_layer")(dense2)

    model = models.Model(inputs=[drug_input, disease_input], outputs=output)
    
    # Log model summary to string buffer
    model_summary_lines = []
    model.summary(print_fn=lambda x: model_summary_lines.append(x))
    logger.info("Model architecture:\n" + "\n".join(model_summary_lines))

    return model


def get_lr_scheduler(scheduler_type='constant', initial_lr=0.002, epochs=1000):
    """
    Create learning rate scheduler based on the specified type.
    
    Args:
        scheduler_type: Type of learning rate scheduler
        initial_lr: Initial learning rate
        epochs: Total number of epochs
        
    Returns:
        Learning rate scheduler or float value
    """
    decay_steps = epochs * 0.8  # Decay for 80% of total epochs
    
    if scheduler_type == 'constant':
        return initial_lr
    
    elif scheduler_type == 'exponential':
        # Exponentially decrease to 1% of initial learning rate
        decay_rate = 0.01 ** (1 / decay_steps)
        return ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False
        )
    
    elif scheduler_type == 'step':
        # Step decay at 50% and 75% of training
        boundaries = [int(epochs * 0.5), int(epochs * 0.75)]
        values = [initial_lr, initial_lr * 0.1, initial_lr * 0.01]
        return PiecewiseConstantDecay(boundaries, values)
    
    elif scheduler_type == 'cosine':
        # Cosine decay to 0 by the end of training
        return CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=epochs,
            alpha=0.0  # Minimum learning rate factor
        )
    
    elif scheduler_type == 'cyclic':
        # Will be implemented with a custom callback
        return initial_lr
    
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using constant")
        return initial_lr


def create_callbacks(lr_scheduler_type='constant', initial_lr=0.002, model_dir="models", epochs=1000):
    """
    Create a list of callbacks for model training.
    
    Args:
        lr_scheduler_type: Type of learning rate scheduler
        initial_lr: Initial learning rate
        model_dir: Directory to save model and logs
        epochs: Total number of epochs
        
    Returns:
        List of callbacks
    """
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = []
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,  # Increased patience with dynamic LR
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)
    
    # Model checkpoint
    model_path = os.path.join(model_dir, "best_model.keras")
    checkpoint = ModelCheckpoint(
        model_path, monitor="val_loss", save_best_only=True
    )
    callbacks.append(checkpoint)
    
    # CSV Logger for detailed metrics tracking
    csv_logger = CSVLogger(os.path.join(model_dir, 'training_log.csv'))
    callbacks.append(csv_logger)
    
    # Dynamic learning rate configuration
    if lr_scheduler_type == 'reduce_on_plateau':
        # Reduce learning rate when validation loss plateaus
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,  # Wait for 5 epochs before reducing
            min_lr=1e-6,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
    elif lr_scheduler_type == 'cyclic':
        # Custom cyclic learning rate
        def cyclic_lr(epoch, lr):
            # Cycle every 10 epochs
            cycle_length = 10
            # Define max and min learning rate
            max_lr = initial_lr
            min_lr = initial_lr / 10
            
            # Calculate where in the cycle we are
            cycle = math.floor(1 + epoch / cycle_length)
            x = epoch - (cycle - 1) * cycle_length
            # Calculate learning rate
            return min_lr + (max_lr - min_lr) * max(0, (cycle_length - x) / cycle_length)
        
        lr_scheduler = LearningRateScheduler(cyclic_lr, verbose=1)
        callbacks.append(lr_scheduler)
        
    elif lr_scheduler_type == 'one_cycle':
        # One-cycle learning rate policy
        def one_cycle_lr(epoch, lr):
            # Parameters
            max_lr = initial_lr * 10
            min_lr = initial_lr / 10
            
            # Divide training into two phases: increase and decrease
            mid_point = epochs // 2
            
            if epoch < mid_point:
                # Linear increase from initial_lr to max_lr
                return initial_lr + (max_lr - initial_lr) * epoch / mid_point
            else:
                # Cosine decay from max_lr to min_lr
                decay_epochs = epochs - mid_point
                t = (epoch - mid_point) / decay_epochs
                return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(t * math.pi))
        
        lr_scheduler = LearningRateScheduler(one_cycle_lr, verbose=1)
        callbacks.append(lr_scheduler)
    
    # Note: For 'constant', 'exponential', 'step', and 'cosine' lr schedulers,
    # the scheduler is built into the optimizer so no callbacks are needed
    
    return callbacks


def model_compilation_training(
    model, drugs, diseases, labels, lr=0.002, epochs=1000, batch_size=16, cnn=False, model_dir="models", 
    lr_scheduler_type='constant', n_splits=5
):
    """
    Compile and train the Siamese Neural Network model with 5-fold cross-validation.
    
    Args:
        model: SNN model to train
        drugs: Drug vectors
        diseases: Disease vectors
        labels: Labels
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Training batch size
        cnn: Whether using CNN architecture
        model_dir: Directory to save model
        lr_scheduler_type: Type of learning rate scheduler
        n_splits: Number of folds for cross-validation
        
    Returns:
        List of training histories for each fold
    """
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    logger.info(f"Compiling model with learning rate {lr} and scheduler {lr_scheduler_type}")
    
    # Prepare CNN input if needed
    if cnn:
        drugs = np.expand_dims(drugs, axis=-1)
        diseases = np.expand_dims(diseases, axis=-1)
    
    # Get learning rate scheduler
    if lr_scheduler_type in ['constant', 'reduce_on_plateau', 'cyclic', 'one_cycle']:
        learning_rate = lr
    else:
        learning_rate = get_lr_scheduler(lr_scheduler_type, lr, epochs)
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy", 
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ],
    )
    
    # Get callbacks
    callbacks = create_callbacks(lr_scheduler_type, lr, model_dir, epochs)
    
    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True)
    
    histories = []

    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kfold.split(drugs, labels)):
        logger.info(f"Training fold {fold + 1}/{n_splits}")
        
        drugs_train, drugs_val = drugs[train_idx], drugs[val_idx]
        diseases_train, diseases_val = diseases[train_idx], diseases[val_idx]
        labels_train, labels_val = labels[train_idx], labels[val_idx]

        history = model.fit(
            [drugs_train, diseases_train],
            labels_train,
            validation_data=([drugs_val, diseases_val], labels_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            # class_weight=class_weight_dict
        )

        histories.append(history)
        
        # Save the model for each fold
        # fold_model_path = os.path.join(model_dir, f"snn_model_fold_{fold + 1}.keras")
        # model.save(fold_model_path)
        # logger.info(f"Model for fold {fold + 1} saved to {fold_model_path}")
        
        # Log training results for each fold
        best_val_accuracy = round(max(history.history['val_accuracy'])*100, 2)
        best_val_loss = round(min(history.history['val_loss']), 2)
        best_val_precision = round(max(history.history['val_precision']), 2)
        best_val_recall = round(max(history.history['val_recall']), 2)
        logger.info(f"Fold {fold + 1} - Best validation accuracy: {best_val_accuracy}%")
        logger.info(f"Fold {fold + 1} - Best validation loss: {best_val_loss}")
        logger.info(f"Fold {fold + 1} - Best validation precision: {best_val_precision}")
        logger.info(f"Fold {fold + 1} - Best validation recall: {best_val_recall}")
    
    return histories
