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
    accuracy_score
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
    
    # Define the list of output files to check
    output_dir = os.path.join(RESULTS_DIR, 'svm_benchmark')
    output_files = [
        os.path.join(output_dir, 'confusion_matrix.png'),
        os.path.join(output_dir, 'true_positives_ids.csv'),
        os.path.join(output_dir, 'false_negatives_ids.csv'),
        os.path.join(output_dir, 'true_negatives_ids.csv'),
        os.path.join(output_dir, 'false_positives_ids.csv'),
        os.path.join(output_dir, 'benchmark_metrics.csv')
    ]
    
    # Check if all output files exist
    if all(os.path.exists(file) for file in output_files):
        logger.info("All benchmarking output files already exist. Skipping SVM benchmarking.")
        print("All benchmarking output files already exist. Skipping SVM benchmarking.")
        return
    else:
        logger.info("One or more benchmarking output files are missing. Proceeding with SVM benchmarking.")
        print("One or more benchmarking output files are missing. Proceeding with SVM benchmarking.")
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)

    # Define paths
    selected_features_file = os.path.join(SELECTED_FEATURES_DIR, 'final_top_20_features.csv')
    test_data_file = TEST_NORM_PROTEIN_FEATURES_FILE
    model_file = os.path.join(RESULTS_DIR, 'final_svm_model.joblib')
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
    test_accuracy = accuracy_score(y_test, y_pred)

    logger.info("=== Benchmark Results ===")
    logger.info(f"MCC: {test_mcc:.4f}")
    logger.info(f"F1 Score: {test_f1:.4f}")
    logger.info(f"Precision: {test_precision:.4f}")
    logger.info(f"Recall: {test_recall:.4f}")
    logger.info(f"Accuracy: {test_accuracy:.4f}")

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

    # Define paths to additional files
    neg_filtered_proteins_file = os.path.join('data', 'fetched_data', 'neg_filtered_proteins.tsv')

    # Check if neg_filtered_proteins.tsv exists
    if not os.path.exists(neg_filtered_proteins_file):
        logger.error(f"Neg filtered proteins file not found: {neg_filtered_proteins_file}")
        return

    # Load neg_filtered_proteins.tsv
    neg_filtered_df = pd.read_csv(neg_filtered_proteins_file, sep='\t')
    logger.info(f"Loaded neg filtered proteins from {neg_filtered_proteins_file}")

    # Merge FP IDs with neg_filtered_df to get 'tm_helix' status
    fp_tm_df = pd.merge(fp_ids_df, neg_filtered_df[['primary_accession', 'tm_helix']], 
                        left_on=expected_id_column, right_on='primary_accession', how='left')
    if fp_tm_df['tm_helix'].isnull().any():
        missing_ids = fp_tm_df[fp_tm_df['tm_helix'].isnull()][expected_id_column].tolist()
        logger.warning(f"The following FP IDs were not found in neg_filtered_proteins.tsv: {missing_ids}")

    # Merge TN IDs with neg_filtered_df to get 'tm_helix' status
    tn_tm_df = pd.merge(tn_ids_df, neg_filtered_df[['primary_accession', 'tm_helix']], 
                        left_on=expected_id_column, right_on='primary_accession', how='left')
    if tn_tm_df['tm_helix'].isnull().any():
        missing_ids = tn_tm_df[tn_tm_df['tm_helix'].isnull()][expected_id_column].tolist()
        logger.warning(f"The following TN IDs were not found in neg_filtered_proteins.tsv: {missing_ids}")

    # Overall FPR
    total_fp = len(fp_ids_df)
    total_tn = len(tn_ids_df)
    overall_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    logger.info(f"Overall False Positive Rate (FPR): {overall_fpr:.4f}")

    # FPR for Transmembrane Proteins
    fp_tm = fp_tm_df['tm_helix'].sum()
    tn_tm = tn_tm_df['tm_helix'].sum()
    fpr_tm = fp_tm / (fp_tm + tn_tm) if (fp_tm + tn_tm) > 0 else 0
    logger.info(f"False Positive Rate for Transmembrane Proteins (FPR_TM): {fpr_tm:.4f}")

    # FPR for Non-Transmembrane Proteins
    fp_non_tm = total_fp - fp_tm
    tn_non_tm = total_tn - tn_tm
    fpr_non_tm = fp_non_tm / (fp_non_tm + tn_non_tm) if (fp_non_tm + tn_non_tm) > 0 else 0
    logger.info(f"False Positive Rate for Non-Transmembrane Proteins (FPR_Non_TM): {fpr_non_tm:.4f}")

    # Save metrics to a file
    metrics = {
        'MCC': [test_mcc],
        'F1_Score': [test_f1],
        'Precision': [test_precision],
        'Recall': [test_recall],
        'Accuracy': [test_accuracy],
        'Overall_FPR': [overall_fpr],
        'FPR_Transmembrane_Proteins': [fpr_tm],
        'FPR_Non_Transmembrane_Proteins': [fpr_non_tm]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_file = os.path.join(output_dir, 'benchmark_metrics.csv')
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Benchmark metrics saved to {metrics_file}")

    logger.info("SVM benchmarking pipeline completed.")
