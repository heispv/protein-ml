# pipeline_12_svm_benchmark.py

import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from config import (
    SELECTED_FEATURES_DIR,
    RESULTS_DIR,
    RANDOM_SEED,
    TEST_NORM_PROTEIN_FEATURES_FILE,
    EXPECTED_ID_COLUMN,
)

logger = logging.getLogger(__name__)

def perform_svm_benchmark():
    """
    Performs benchmarking of the trained SVM model on the test dataset.
    """
    logger.info("Starting SVM benchmarking pipeline.")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Define paths
    selected_features_file = os.path.join(SELECTED_FEATURES_DIR, 'final_top_20_features.csv')
    test_data_file = TEST_NORM_PROTEIN_FEATURES_FILE
    model_file = os.path.join(RESULTS_DIR, 'final_svm_model.joblib')
    output_dir = os.path.join(RESULTS_DIR, 'svm_benchmark')
    os.makedirs(output_dir, exist_ok=True)

    # Load selected features
    if not os.path.exists(selected_features_file):
        logger.error(f"Selected features file not found: {selected_features_file}")
        return

    selected_features_df = pd.read_csv(selected_features_file)
    selected_features = selected_features_df['feature'].tolist()
    logger.info(f"Loaded selected features from {selected_features_file}")

    # Load test dataset
    if not os.path.exists(test_data_file):
        logger.error(f"Test data file not found: {test_data_file}")
        return

    test_df = pd.read_csv(test_data_file)
    logger.info(f"Loaded test data from {test_data_file}")

    # Verify presence of accession_id column
    expected_id_column = EXPECTED_ID_COLUMN  # Defined in config.py
    if expected_id_column not in test_df.columns:
        available_columns = ', '.join(test_df.columns)
        logger.error(
            f"Expected column '{expected_id_column}' not found in test data. "
            f"Available columns are: {available_columns}"
        )
        return

    # Ensure that selected features are in the test dataframe
    missing_features = [feature for feature in selected_features if feature not in test_df.columns]
    if missing_features:
        logger.error(f"The following selected features are missing in the test dataset: {missing_features}")
        return

    # Load the trained SVM model
    if not os.path.exists(model_file):
        logger.error(f"SVM model file not found: {model_file}")
        return

    svm_model = joblib.load(model_file)
    logger.info(f"Loaded trained SVM model from {model_file}")

    # Prepare test data
    X_test = test_df[selected_features]
    y_test = test_df['label']
    accession_ids = test_df[expected_id_column]

    # Predict on test data
    y_pred = svm_model.predict(X_test)

    # Calculate metrics
    test_mcc = matthews_corrcoef(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)

    logger.info("=== Benchmark Results ===")
    logger.info(f"MCC: {test_mcc:.4f}")
    logger.info(f"F1 Score: {test_f1:.4f}")
    logger.info(f"Precision: {test_precision:.4f}")
    logger.info(f"Recall: {test_recall:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_file)
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_file}")

    # Extract IDs of True Positives (TP) and False Negatives (FN)
    tp_indices = (y_test == 1) & (y_pred == 1)
    fn_indices = (y_test == 1) & (y_pred == 0)

    tp_ids = accession_ids[tp_indices]
    fn_ids = accession_ids[fn_indices]

    # Save TP IDs
    tp_file = os.path.join(output_dir, 'true_positives_ids.csv')
    tp_ids_df = pd.DataFrame({expected_id_column: tp_ids})
    tp_ids_df.to_csv(tp_file, index=False)
    logger.info(f"True Positive IDs saved to {tp_file}")

    # Save FN IDs
    fn_file = os.path.join(output_dir, 'false_negatives_ids.csv')
    fn_ids_df = pd.DataFrame({expected_id_column: fn_ids})
    fn_ids_df.to_csv(fn_file, index=False)
    logger.info(f"False Negative IDs saved to {fn_file}")

    # Extract IDs of True Negatives (TN) and False Positives (FP)
    tn_indices = (y_test == 0) & (y_pred == 0)
    fp_indices = (y_test == 0) & (y_pred == 1)

    tn_ids = accession_ids[tn_indices]
    fp_ids = accession_ids[fp_indices]

    # Save TN IDs
    tn_file = os.path.join(output_dir, 'true_negatives_ids.csv')
    tn_ids_df = pd.DataFrame({expected_id_column: tn_ids})
    tn_ids_df.to_csv(tn_file, index=False)
    logger.info(f"True Negative IDs saved to {tn_file}")

    # Save FP IDs
    fp_file = os.path.join(output_dir, 'false_positives_ids.csv')
    fp_ids_df = pd.DataFrame({expected_id_column: fp_ids})
    fp_ids_df.to_csv(fp_file, index=False)
    logger.info(f"False Positive IDs saved to {fp_file}")

    # Save metrics to a file
    metrics = {
        'MCC': [test_mcc],
        'F1_Score': [test_f1],
        'Precision': [test_precision],
        'Recall': [test_recall]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_file = os.path.join(output_dir, 'benchmark_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Benchmark metrics saved to {metrics_file}")

    logger.info("SVM benchmarking pipeline completed.")
