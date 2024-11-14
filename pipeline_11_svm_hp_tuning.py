# pipeline_11_svm_hp_tuning.py

import os
import logging
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict
import joblib  # For saving the trained model
from config import (
    SELECTED_FEATURES_DIR,
    NUM_FOLDS,
    RANDOM_SEED,
    RESULTS_DIR,
    N_JOBS,
    NORM_PROTEIN_FEATURES_FILE,
)

logger = logging.getLogger(__name__)

def perform_svm_hyperparameter_tuning():
    """
    Performs hyperparameter tuning for SVM using cross-validation.
    """
    logger.info("Starting SVM hyperparameter tuning pipeline.")
    
    # Define the list of output files to check
    output_files = [
        os.path.join(RESULTS_DIR, 'final_svm_model.joblib'),
        os.path.join(RESULTS_DIR, 'svm_initial_5fold_results.csv'),
        os.path.join(RESULTS_DIR, 'svm_refined_hyperparameters.csv')
    ]
    
    # Check if all output files exist
    if all(os.path.exists(file) for file in output_files):
        logger.info("All hyperparameter tuning output files already exist. Skipping hyperparameter tuning.")
        print("All hyperparameter tuning output files already exist. Skipping hyperparameter tuning.")
        return
    else:
        logger.info("One or more hyperparameter tuning output files are missing. Proceeding with hyperparameter tuning.")
        print("One or more hyperparameter tuning output files are missing. Proceeding with hyperparameter tuning.")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    # Load the dataset
    data_file = NORM_PROTEIN_FEATURES_FILE
    if not os.path.exists(data_file):
        logger.error(f"Data file not found: {data_file}")
        return
    df = pd.read_csv(data_file)
    logger.info(f"Loaded data from {data_file}")
    
    # Load selected features
    selected_features_file = os.path.join(SELECTED_FEATURES_DIR, 'final_top_20_features.csv')
    if not os.path.exists(selected_features_file):
        logger.error(f"Selected features file not found: {selected_features_file}")
        return
    selected_features_df = pd.read_csv(selected_features_file)
    selected_features = selected_features_df['feature'].tolist()
    logger.info(f"Loaded selected features from {selected_features_file}")
    
    # Ensure that selected features are in the dataframe
    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        logger.error(f"Features not found in the dataset: {missing_features}")
        return
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    logger.info("Data shuffled")
    
    # Assign fold numbers to each sample
    df['fold'] = df.index % NUM_FOLDS
    logger.info(f"Assigned fold numbers with {NUM_FOLDS} folds")
    
    # Prepare data
    X = df[selected_features]
    y = df['label']
    
    # Store results
    results = []
    best_hyperparams = {'C': [], 'gamma': []}
    
    # Define initial hyperparameter grid
    C_values = np.linspace(1, 10, 8)
    gamma_values = [0.01, 0.03, 0.065, 0.1, 0.13, 0.165, 0.2, 0.25, 0.3]
    
    logger.info("=== Step 1: Initial 5-Fold Hyperparameter Tuning ===")
    for i in range(NUM_FOLDS):
        # Define fold numbers for training, validation, and testing
        train_folds = [(i + j) % NUM_FOLDS for j in range(3)]  # Folds i, i+1, i+2
        val_fold = (i + 3) % NUM_FOLDS
        test_fold = (i + 4) % NUM_FOLDS
        
        logger.info(f"Processing fold {i + 1}: Training folds {train_folds}, Validation fold {val_fold}, Test fold {test_fold}")
        
        # Split the data
        train_idx = df[df['fold'].isin(train_folds)].index
        val_idx = df[df['fold'] == val_fold].index
        test_idx = df[df['fold'] == test_fold].index
        
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_val = X.loc[val_idx]
        y_val = y.loc[val_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        
        best_mcc = -1
        best_C = None
        best_gamma = None
        
        # Grid search over hyperparameters
        for C in C_values:
            for gamma in gamma_values:
                # Create the SVM model
                svm = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
                
                # Train on training set
                svm.fit(X_train, y_train)
                
                # Predict on validation set
                y_pred_val = svm.predict(X_val)
                
                # Compute MCC
                mcc = matthews_corrcoef(y_val, y_pred_val)
                
                # Update best hyperparameters
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_C = C
                    best_gamma = gamma
        
        # Store the best hyperparameters
        best_hyperparams['C'].append(best_C)
        best_hyperparams['gamma'].append(best_gamma)
        logger.info(f"Fold {i + 1}: Best C={best_C}, gamma={best_gamma}, Validation MCC={best_mcc:.4f}")
        
        # Train final model with best hyperparameters on training set
        svm = SVC(C=best_C, gamma=best_gamma, kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
        svm.fit(X_train, y_train)
        
        # Predict on test set
        y_pred_test = svm.predict(X_test)
        
        # Compute metrics on test set
        test_mcc = matthews_corrcoef(y_test, y_pred_test)
        test_precision = precision_score(y_test, y_pred_test, zero_division=0)
        test_recall = recall_score(y_test, y_pred_test, zero_division=0)
        
        # Collect results
        result = {
            'fold': i + 1,
            'best_C': best_C,
            'best_gamma': best_gamma,
            'validation_mcc': best_mcc,
            'test_mcc': test_mcc,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
        results.append(result)
        logger.info(f"Fold {i + 1}: Test MCC={test_mcc:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Determine the min and max of best C and gamma
    min_C = min(best_hyperparams['C'])
    max_C = max(best_hyperparams['C'])
    min_gamma = min(best_hyperparams['gamma'])
    max_gamma = max(best_hyperparams['gamma'])
    
    logger.info("=== Step 2: Refined Hyperparameter Tuning on Entire Training Dataset ===")
    logger.info(f"Refined C range: {min_C} to {max_C}")
    logger.info(f"Refined gamma range: {min_gamma} to {max_gamma}")
    
    # Define refined hyperparameter grid
    refined_C_values = np.linspace(min_C, max_C, 8)
    refined_gamma_values = np.linspace(min_gamma, max_gamma, 8)
    
    # Initialize variables to store the best hyperparameters from the refined grid search
    final_best_mcc = -1
    final_best_C = None
    final_best_gamma = None
    
    # Perform grid search on the entire dataset
    for C in refined_C_values:
        for gamma in refined_gamma_values:
            # Create the SVM model
            svm = SVC(C=C, gamma=gamma, kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
            
            # Train on the entire dataset
            svm.fit(X, y)
            
            # Perform cross-validation to evaluate MCC
            y_pred = cross_val_predict(svm, X, y, cv=NUM_FOLDS, n_jobs=N_JOBS)
            mcc = matthews_corrcoef(y, y_pred)
            
            # Update best hyperparameters
            if mcc > final_best_mcc:
                final_best_mcc = mcc
                final_best_C = C
                final_best_gamma = gamma
    
    logger.info(f"Refined Best C={final_best_C}, gamma={final_best_gamma}, Cross-Validated MCC={final_best_mcc:.4f}")
    
    # Initialize variables to store additional metrics
    final_precision = precision_score(y, y_pred, zero_division=0)
    final_recall = recall_score(y, y_pred, zero_division=0)
    final_f1 = f1_score(y, y_pred, zero_division=0)
    final_accuracy = accuracy_score(y, y_pred)
    
    logger.info(f"Refined Cross-Validated Precision={final_precision:.4f}, Recall={final_recall:.4f}, F1 Score={final_f1:.4f}, Accuracy={final_accuracy:.4f}")
    
    # Train final model with refined hyperparameters on the entire dataset
    final_svm = SVC(C=final_best_C, gamma=final_best_gamma, kernel='rbf', class_weight='balanced', random_state=RANDOM_SEED)
    final_svm.fit(X, y)
    
    # Save the final trained SVM model for later use
    output_dir = RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    model_file = os.path.join(output_dir, 'final_svm_model.joblib')
    joblib.dump(final_svm, model_file)
    logger.info(f"Final trained SVM model saved to {model_file}")
    
    # Save all results to CSV
    # Save initial 5-fold results
    initial_results_file = os.path.join(output_dir, 'svm_initial_5fold_results.csv')
    results_df.to_csv(initial_results_file, index=False)
    logger.info(f"Initial 5-Fold Results saved to {initial_results_file}")
    
    # Save final refined hyperparameters and additional metrics
    refined_results = {
        'final_best_C': final_best_C,
        'final_best_gamma': final_best_gamma,
        'final_cross_validated_MCC': final_best_mcc,
        'final_cross_validated_Precision': final_precision,
        'final_cross_validated_Recall': final_recall,
        'final_cross_validated_F1_Score': final_f1,
        'final_cross_validated_Accuracy': final_accuracy
    }
    refined_results_df = pd.DataFrame([refined_results])
    refined_results_file = os.path.join(output_dir, 'svm_refined_hyperparameters.csv')
    refined_results_df.to_csv(refined_results_file, index=False)
    logger.info(f"Refined Hyperparameters and Metrics saved to {refined_results_file}")
    
    logger.info("SVM hyperparameter tuning pipeline completed.")
