import pandas as pd
import numpy as np
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
from pathlib import Path
import sys

def main():
    # Set random seed for reproducibility
    random_seed = 42
    np.random.seed(random_seed)

    # Define paths
    selected_features_file = Path('data/features/selected_features/final_top_20_features.csv')
    test_data_file = Path('data/features/testing/test_norm_protein_features.csv')
    model_file = Path('data/results/final_svm_model.joblib')
    refined_hyperparams_file = Path('data/results/svm_refined_hyperparameters.csv')
    output_dir = Path('data/results/svm_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load selected features
    if not selected_features_file.exists():
        raise FileNotFoundError(f"Selected features file not found: {selected_features_file}")

    selected_features_df = pd.read_csv(selected_features_file)
    selected_features = selected_features_df['feature'].tolist()

    # Load test dataset
    if not test_data_file.exists():
        raise FileNotFoundError(f"Test data file not found: {test_data_file}")

    test_df = pd.read_csv(test_data_file)

    # Verify presence of accession_id column
    expected_id_column = 'accession_id'  # Update this if your column has a different name
    if expected_id_column not in test_df.columns:
        available_columns = ', '.join(test_df.columns)
        raise KeyError(
            f"Expected column '{expected_id_column}' not found in test data. "
            f"Available columns are: {available_columns}"
        )

    # Ensure that selected features are in the test dataframe
    missing_features = [feature for feature in selected_features if feature not in test_df.columns]
    if missing_features:
        raise ValueError(f"The following selected features are missing in the test dataset: {missing_features}")

    # Load the trained SVM model
    if not model_file.exists():
        raise FileNotFoundError(f"SVM model file not found: {model_file}")

    svm_model = joblib.load(model_file)
    print(f"Loaded trained SVM model from {model_file}")

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

    print("=== Benchmark Results ===")
    print(f"MCC: {test_mcc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_file = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_file)
    plt.close()
    print(f"Confusion matrix saved to {cm_file}")

    # Extract IDs of True Positives (TP) and False Negatives (FN)
    tp_indices = (y_test == 1) & (y_pred == 1)
    fn_indices = (y_test == 1) & (y_pred == 0)

    tp_ids = accession_ids[tp_indices]
    fn_ids = accession_ids[fn_indices]

    # Save TP IDs
    tp_file = output_dir / 'true_positives_ids.csv'
    tp_ids_df = pd.DataFrame({expected_id_column: tp_ids})
    tp_ids_df.to_csv(tp_file, index=False)
    print(f"True Positive IDs saved to {tp_file}")

    # Save FN IDs
    fn_file = output_dir / 'false_negatives_ids.csv'
    fn_ids_df = pd.DataFrame({expected_id_column: fn_ids})
    fn_ids_df.to_csv(fn_file, index=False)
    print(f"False Negative IDs saved to {fn_file}")

    # Save metrics to a file
    metrics = {
        'MCC': [test_mcc],
        'F1_Score': [test_f1],
        'Precision': [test_precision],
        'Recall': [test_recall]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_file = output_dir / 'benchmark_metrics.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Benchmark metrics saved to {metrics_file}")

    print("=== Benchmarking Completed ===")

if __name__ == '__main__':
    try:
        main()
    except KeyError as e:
        print(f"KeyError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
