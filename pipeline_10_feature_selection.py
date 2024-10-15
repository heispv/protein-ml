# pipeline_10_feature_selection.py

import os
import logging
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from config import NUM_FOLDS, RANDOM_SEED, SELECTED_FEATURES_DIR, PROTEIN_FEATURES_FILE

logger = logging.getLogger(__name__)

def perform_feature_selection():
    """
    Performs feature selection using Random Forest and cross-validation.
    """
    logger.info("Starting feature selection pipeline.")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Load the dataset
    data_file = PROTEIN_FEATURES_FILE
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    df = pd.read_csv(data_file)
    logger.info(f"Loaded data from {data_file}")
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    logger.info("Data shuffled")
    
    # Assign fold numbers to each sample
    df['fold'] = df.index % NUM_FOLDS
    logger.info(f"Assigned fold numbers with {NUM_FOLDS} folds")
    
    # Get feature columns (excluding 'accession_id', 'label', 'fold')
    feature_columns = [col for col in df.columns if col not in ['accession_id', 'label', 'fold']]
    
    # Store selected features from each fold
    fold_selected_features = []
    
    # Perform cross-validation
    for i in range(NUM_FOLDS):
        # Define fold numbers for training, validation, and testing
        train_folds = [(i + j) % NUM_FOLDS for j in range(3)]  # Folds i, i+1, i+2
        val_fold = (i + 3) % NUM_FOLDS
        test_fold = (i + 4) % NUM_FOLDS

        logger.info(f"Processing fold {i + 1}: Training folds {train_folds}, Validation fold {val_fold}, Test fold {test_fold}")
        
        # Split the data
        train_data = df[df['fold'].isin(train_folds)]
        val_data = df[df['fold'] == val_fold]
        test_data = df[df['fold'] == test_fold]

        X_train = train_data[feature_columns]
        y_train = train_data['label']
        X_val = val_data[feature_columns]
        y_val = val_data['label']
        X_test = test_data[feature_columns]
        y_test = test_data['label']

        # Train the Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
        rf.fit(X_train, y_train)
        logger.info(f"Random Forest model trained for fold {i + 1}")

        # Get feature importances
        importances = rf.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': importances
        })

        # Select top 20 features
        top_features = feature_importance_df.sort_values(by='importance', ascending=False).head(20)
        selected_features = top_features['feature'].tolist()
        fold_selected_features.append(selected_features)
        logger.info(f"Top 20 features selected for fold {i + 1}")

        # Save the feature importances for this fold
        output_dir = SELECTED_FEATURES_DIR
        os.makedirs(output_dir, exist_ok=True)
        fold_num = i + 1  # Folds numbered from 1 to NUM_FOLDS
        output_file = os.path.join(output_dir, f'fold_{fold_num}_feature_importances.csv')
        feature_importance_df.to_csv(output_file, index=False)
        logger.info(f"Fold {fold_num}: Feature importances saved to {output_file}")

    # Aggregate selected features from all folds
    all_selected_features = [feature for fold_features in fold_selected_features for feature in fold_features]
    feature_counter = Counter(all_selected_features)

    # Get the most common features across all folds
    common_features = feature_counter.most_common(20)
    final_selected_features = [feature for feature, count in common_features]

    # Save the final selected features
    final_features_df = pd.DataFrame(common_features, columns=['feature', 'count'])
    final_output_file = os.path.join(output_dir, 'final_top_20_features.csv')
    final_features_df.to_csv(final_output_file, index=False)
    logger.info(f"Final top 20 selected features saved to {final_output_file}")
    
    logger.info("Feature selection pipeline completed.")
