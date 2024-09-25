# pipeline_08_vonHeijne_benchmark.py

import os
import numpy as np
import pandas as pd
import logging
from Bio import SeqIO
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from typing import List, Dict, Tuple, Optional

# Define amino acid to index mapping for one-hot encoding
AMINO_ACIDS = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G',
    'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S',
    'T', 'W', 'Y', 'V'
]
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# SwissProt background frequencies
SWISSPROT_FREQ = {
    'A': 0.08, 'R': 0.06, 'N': 0.04, 'D': 0.06,
    'C': 0.01, 'Q': 0.04, 'E': 0.07, 'G': 0.07,
    'H': 0.02, 'I': 0.06, 'L': 0.10, 'K': 0.06,
    'M': 0.02, 'F': 0.04, 'P': 0.05, 'S': 0.07,
    'T': 0.05, 'W': 0.01, 'Y': 0.03, 'V': 0.07
}

def compute_train_counts_from_file(fasta_file_path: str, aa_to_index: Dict[str, int]) -> Tuple[np.ndarray, int]:
    train_counts = np.zeros((20, 15), dtype=float)
    total_sequences = 0

    if not os.path.isfile(fasta_file_path):
        logging.error(f"Training fasta file {fasta_file_path} does not exist.")
        return train_counts, total_sequences

    try:
        for record in SeqIO.parse(fasta_file_path, 'fasta'):
            seq = str(record.seq).upper()
            if len(seq) != 15:
                logging.warning(f"Sequence length {len(seq)} != 15 in {fasta_file_path}. Skipping sequence {record.id}.")
                continue
            for pos, aa in enumerate(seq):
                if aa == 'U':
                    aa = 'C'
                    logging.info(f"Changed U to C in sequence {record.id} at position {pos}.")
                if aa in aa_to_index:
                    train_counts[aa_to_index[aa], pos] += 1
                else:
                    logging.warning(f"Unknown amino acid '{aa}' in sequence {record.id} at position {pos}. Skipping.")
            total_sequences += 1
    except Exception as e:
        logging.error(f"Error reading fasta file {fasta_file_path}: {e}")
    
    return train_counts, total_sequences

def compute_scoring_matrix(train_counts: np.ndarray, total_sequences: int) -> Optional[np.ndarray]:
    if total_sequences == 0:
        logging.error("No valid training sequences found. Cannot compute scoring matrix.")
        return None

    train_counts += 1  # Add pseudocount
    train_counts /= (20 + total_sequences)  # Normalize counts

    # Divide by SwissProt frequencies
    for aa, freq in SWISSPROT_FREQ.items():
        if aa in AA_TO_INDEX:
            idx = AA_TO_INDEX[aa]
            train_counts[idx, :] /= freq
        else:
            logging.warning(f"Amino acid '{aa}' not found in mapping. Skipping frequency division.")

    # Take natural logarithm
    with np.errstate(divide='ignore'):
        scoring_matrix = np.log(train_counts)

    return scoring_matrix

def read_threshold_from_file(threshold_file: str) -> Optional[float]:
    if not os.path.isfile(threshold_file):
        logging.error(f"Threshold file {threshold_file} does not exist.")
        return None
    try:
        with open(threshold_file, 'r') as f:
            for line in f:
                if 'Final Threshold' in line:
                    # Extract the threshold value
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        threshold_value = float(parts[1].strip())
                        logging.info(f"Threshold read from file: {threshold_value}")
                        return threshold_value
        logging.error(f"Could not find 'Final Threshold' in file {threshold_file}")
    except Exception as e:
        logging.error(f"Error reading threshold from file {threshold_file}: {e}")
    return None

def read_sequences_from_dir(dir_path: str) -> List[str]:
    sequences = []
    if not os.path.isdir(dir_path):
        logging.warning(f"Directory {dir_path} does not exist. Skipping.")
        return sequences
    fasta_files = [f for f in os.listdir(dir_path) if f.endswith('.fasta')]
    if not fasta_files:
        logging.warning(f"No .fasta files found in {dir_path}. Skipping.")
        return sequences
    for fasta_file in fasta_files:
        fasta_path = os.path.join(dir_path, fasta_file)
        try:
            for record in SeqIO.parse(fasta_path, 'fasta'):
                seq = str(record.seq).upper().replace('U', 'C')
                sequences.append(seq)
        except Exception as e:
            logging.error(f"Error reading fasta file {fasta_path}: {e}")
    return sequences

def compute_sequence_scores(sequences: List[str], scoring_matrix: np.ndarray) -> List[float]:
    scores = []
    for seq in sequences:
        if len(seq) < 15:
            logging.warning(f"Sequence length {len(seq)} < 15. Skipping sequence.")
            continue
        if len(seq) > 90:
            seq = seq[:90]
        window_scores = []
        for i in range(len(seq) - 14):
            window = seq[i:i + 15]
            score = 0.0
            valid_window = True
            for pos, aa in enumerate(window):
                if aa in AA_TO_INDEX:
                    score += scoring_matrix[AA_TO_INDEX[aa], pos]
                else:
                    logging.warning(f"Unknown amino acid '{aa}' at position {i + pos} in sequence. Skipping window.")
                    valid_window = False
                    break
            if valid_window:
                window_scores.append(score)
        if window_scores:
            max_score = max(window_scores)
            scores.append(max_score)
    return scores

def evaluate_predictions(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    mcc = matthews_corrcoef(labels, predictions)
    return {'Precision': precision, 'Recall': recall, 'F1': f1, 'MCC': mcc}

def perform_vonHeijne_benchmark_analysis(
    cleavage_site_seqs_file: str,
    splitted_data_dir: str,
    threshold_file: str,
    output_dir: str
) -> None:
    """
    Performs Von Heijne benchmark analysis using the given cleavage site sequences,
    threshold, and test data.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logging
    log_file = os.path.join(output_dir, 'vonHeijne_benchmark_analysis.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Starting Von Heijne benchmark analysis.")

    # Define output file paths
    scoring_matrix_csv = os.path.join(output_dir, 'scoring_matrix.csv')
    results_csv = os.path.join(output_dir, 'benchmark_results.csv')
    detailed_results_csv = os.path.join(output_dir, 'detailed_results.csv')

    # Check if results already exist
    if os.path.exists(scoring_matrix_csv) and os.path.exists(results_csv) and os.path.exists(detailed_results_csv):
        logging.info("Benchmark analysis files already exist. Skipping computation.")
        print("Benchmark analysis files already exist. Skipping computation.")
        return

    # Training Phase
    logging.info("Training phase started.")
    train_counts, total_sequences = compute_train_counts_from_file(cleavage_site_seqs_file, AA_TO_INDEX)
    if total_sequences == 0:
        logging.error("No valid training sequences found. Cannot compute scoring matrix.")
        return

    scoring_matrix = compute_scoring_matrix(train_counts, total_sequences)
    if scoring_matrix is None:
        logging.error("Scoring matrix computation failed.")
        return
    logging.info("Training phase completed.")

    # Save Scoring Matrix
    scoring_matrix_df = pd.DataFrame(
        scoring_matrix,
        index=AMINO_ACIDS,
        columns=[f'Pos_{i + 1}' for i in range(15)]
    )
    csv_filename = 'scoring_matrix.csv'
    csv_path = os.path.join(output_dir, csv_filename)

    try:
        scoring_matrix_df.to_csv(csv_path)
        logging.info(f"Scoring matrix saved to {csv_path}")
    except Exception as e:
        logging.error(f"Error saving scoring matrix to CSV file {csv_path}: {e}")

    # Read Threshold
    logging.info("Reading threshold from file.")
    threshold = read_threshold_from_file(threshold_file)
    if threshold is None:
        logging.error("Threshold reading failed.")
        return

    # Testing Phase
    logging.info("Testing phase started.")
    test_pos_dir = os.path.join(splitted_data_dir, 'pos')
    test_neg_dir = os.path.join(splitted_data_dir, 'neg')

    # Read test sequences
    test_pos_sequences = read_sequences_from_dir(test_pos_dir)
    test_neg_sequences = read_sequences_from_dir(test_neg_dir)

    if not test_pos_sequences and not test_neg_sequences:
        logging.error("No valid test sequences found.")
        return

    # Compute scores
    test_pos_scores = compute_sequence_scores(test_pos_sequences, scoring_matrix)
    test_neg_scores = compute_sequence_scores(test_neg_sequences, scoring_matrix)

    test_scores = test_pos_scores + test_neg_scores
    test_labels = [1] * len(test_pos_scores) + [0] * len(test_neg_scores)
    test_predictions = [1 if score >= threshold else 0 for score in test_scores]

    # Evaluate
    test_results = evaluate_predictions(test_predictions, test_labels)
    logging.info(
        f"Testing phase completed. Threshold: {threshold}, Precision: {test_results['Precision']}, "
        f"Recall: {test_results['Recall']}, F1: {test_results['F1']}, MCC: {test_results['MCC']}"
    )

    # Save results
    results_df = pd.DataFrame({
        'Threshold': [threshold],
        'Test Precision': [test_results['Precision']],
        'Test Recall': [test_results['Recall']],
        'Test F1': [test_results['F1']],
        'Test MCC': [test_results['MCC']]
    })
    results_csv = os.path.join(output_dir, 'benchmark_results.csv')
    try:
        results_df.to_csv(results_csv, index=False)
        logging.info(f"Results saved to {results_csv}")
    except Exception as e:
        logging.error(f"Error saving results to CSV file {results_csv}: {e}")

    # Save detailed results
    detailed_results_df = pd.DataFrame({
        'Score': test_scores,
        'Label': test_labels,
        'Prediction': test_predictions
    })
    detailed_results_csv = os.path.join(output_dir, 'detailed_results.csv')
    try:
        detailed_results_df.to_csv(detailed_results_csv, index=False)
        logging.info(f"Detailed results saved to {detailed_results_csv}")
    except Exception as e:
        logging.error(f"Error saving detailed results to CSV file {detailed_results_csv}: {e}")

    logging.info("Von Heijne benchmark analysis completed.")
