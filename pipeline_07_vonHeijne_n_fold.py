# pipeline_07_vonHeijne.py

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

# Cross-validation combinations
COMBINATIONS = [
    (['1', '2', '3'], '4', '5'),
    (['2', '3', '4'], '5', '1'),
    (['3', '4', '5'], '1', '2'),
    (['4', '5', '1'], '2', '3'),
    (['5', '1', '2'], '3', '4')
]


def compute_train_counts(pos_dirs: List[str], aa_to_index: Dict[str, int]) -> Tuple[np.ndarray, int]:
    train_counts = np.zeros((20, 15), dtype=float)
    total_sequences = 0

    for pos_dir in pos_dirs:
        if not os.path.isdir(pos_dir):
            logging.warning(f"Training pos directory {pos_dir} does not exist. Skipping.")
            continue
        fasta_files = [f for f in os.listdir(pos_dir) if f.endswith('.fasta')]
        if not fasta_files:
            logging.warning(f"No .fasta files found in {pos_dir}. Skipping.")
            continue
        for fasta_file in fasta_files:
            fasta_path = os.path.join(pos_dir, fasta_file)
            try:
                for record in SeqIO.parse(fasta_path, 'fasta'):
                    seq = str(record.seq).upper()
                    if len(seq) != 15:
                        logging.warning(f"Sequence length {len(seq)} != 15 in {fasta_path}. Skipping sequence {record.id}.")
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
                logging.error(f"Error reading fasta file {fasta_path}: {e}")
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


def find_best_threshold(scores: List[float], labels: List[int]) -> Tuple[Optional[float], Optional[float]]:
    if not scores:
        logging.error("No scores provided for threshold determination.")
        return None, None

    thresholds = sorted(set(scores))
    best_f1 = -1
    best_threshold = None

    for threshold in thresholds:
        predictions = [1 if score >= threshold else 0 for score in scores]
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1


def evaluate_predictions(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    mcc = matthews_corrcoef(labels, predictions)
    return {'Precision': precision, 'Recall': recall, 'F1': f1, 'MCC': mcc}


def perform_vonHeijne_analysis(cleavage_site_seqs_dir: str, splitted_data_dir: str, output_dir: str) -> None:
    """
    Performs the Von Heijne analysis to determine the optimal threshold for classifying
    sequences as having a cleavage site or not based on a scoring matrix derived from training data.

    Additionally, saves the scoring matrix for each training combination as a CSV file,
    and saves the thresholds, F1 scores, precision, recall, MCC, etc., into a results.csv file.

    Args:
        cleavage_site_seqs_dir (str): Path to data/cleavage_site_seqs/train/{1,2,3,4,5}/pos directories.
        splitted_data_dir (str): Path to data/splited_data/train/{1,2,3,4,5}/{pos, neg}/ directories.
        output_dir (str): Path to data/vonHeijne_results/ directory where results will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logging
    log_file = os.path.join(output_dir, 'vonHeijne_analysis.log')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Starting Von Heijne analysis.")

    # Check if the main output files already exist
    final_threshold_file = os.path.join(output_dir, 'final_threshold.txt')
    individual_thresholds_csv = os.path.join(output_dir, 'individual_thresholds.csv')
    results_csv = os.path.join(output_dir, 'results.csv')

    if os.path.exists(final_threshold_file) and os.path.exists(individual_thresholds_csv) and os.path.exists(results_csv):
        logging.info("Von Heijne analysis output files already exist. Skipping computation.")
        print("Von Heijne analysis output files already exist. Skipping computation.")
        return

    final_thresholds = []
    results_list = []

    for idx, (train_folds, validate_fold, test_fold) in enumerate(COMBINATIONS, start=1):
        logging.info(f"Processing combination {idx}: Train folds {train_folds}, Validate fold {validate_fold}, Test fold {test_fold}")

        # Training Phase
        logging.info("Training phase started.")
        pos_dirs = [os.path.join(cleavage_site_seqs_dir, fold, 'pos') for fold in train_folds]
        train_counts, total_sequences = compute_train_counts(pos_dirs, AA_TO_INDEX)

        if total_sequences == 0:
            logging.error(f"No valid training sequences found for combination {idx}. Skipping to next combination.")
            continue

        scoring_matrix = compute_scoring_matrix(train_counts, total_sequences)

        if scoring_matrix is None:
            logging.error(f"Scoring matrix computation failed for combination {idx}. Skipping to next combination.")
            continue

        logging.info("Training phase completed.")

        # Save Scoring Matrix
        scoring_matrix_df = pd.DataFrame(
            scoring_matrix,
            index=AMINO_ACIDS,
            columns=[f'Pos_{i + 1}' for i in range(15)]
        )
        train_folds_str = '_'.join(train_folds)
        csv_filename = f'scoring_matrix_train_folds_{train_folds_str}.csv'
        csv_path = os.path.join(output_dir, csv_filename)

        # Check if scoring matrix already exists
        if os.path.exists(csv_path):
            logging.info(f"Scoring matrix {csv_path} already exists. Skipping saving.")
        else:
            try:
                scoring_matrix_df.to_csv(csv_path)
                logging.info(f"Scoring matrix saved to {csv_path}")
            except Exception as e:
                logging.error(f"Error saving scoring matrix to CSV file {csv_path}: {e}")

        # Validation Phase
        logging.info("Validation phase started.")
        validate_pos_dir = os.path.join(splitted_data_dir, 'train', validate_fold, 'pos')
        validate_neg_dir = os.path.join(splitted_data_dir, 'train', validate_fold, 'neg')

        # Read validation sequences
        validate_pos_sequences = read_sequences_from_dir(validate_pos_dir)
        validate_neg_sequences = read_sequences_from_dir(validate_neg_dir)

        # Compute scores
        validate_pos_scores = compute_sequence_scores(validate_pos_sequences, scoring_matrix)
        validate_neg_scores = compute_sequence_scores(validate_neg_sequences, scoring_matrix)

        validate_scores = validate_pos_scores + validate_neg_scores
        validate_labels = [1] * len(validate_pos_scores) + [0] * len(validate_neg_scores)

        if not validate_scores:
            logging.error(f"No valid validation sequences found for combination {idx}. Skipping to next combination.")
            continue

        # Find best threshold
        best_threshold, best_f1 = find_best_threshold(validate_scores, validate_labels)
        if best_threshold is None:
            logging.error(f"Threshold determination failed for combination {idx}. Skipping to next combination.")
            continue

        logging.info(f"Validation phase completed. Best Threshold: {best_threshold}, Best F1: {best_f1}")

        final_thresholds.append(best_threshold)

        # Testing Phase
        logging.info("Testing phase started.")
        test_pos_dir = os.path.join(splitted_data_dir, 'train', test_fold, 'pos')
        test_neg_dir = os.path.join(splitted_data_dir, 'train', test_fold, 'neg')

        # Read test sequences
        test_pos_sequences = read_sequences_from_dir(test_pos_dir)
        test_neg_sequences = read_sequences_from_dir(test_neg_dir)

        # Compute scores
        test_pos_scores = compute_sequence_scores(test_pos_sequences, scoring_matrix)
        test_neg_scores = compute_sequence_scores(test_neg_sequences, scoring_matrix)

        test_scores = test_pos_scores + test_neg_scores
        test_labels = [1] * len(test_pos_scores) + [0] * len(test_neg_scores)

        if not test_scores:
            logging.error(f"No valid testing sequences found for combination {idx}. Skipping to next combination.")
            continue

        # Apply threshold
        test_predictions = [1 if score >= best_threshold else 0 for score in test_scores]

        # Evaluate
        test_results = evaluate_predictions(test_predictions, test_labels)
        logging.info(
            f"Testing phase completed. Threshold: {best_threshold}, Precision: {test_results['Precision']}, "
            f"Recall: {test_results['Recall']}, F1: {test_results['F1']}, MCC: {test_results['MCC']}"
        )

        # Collect results
        result = {
            'Combination': idx,
            'Train Folds': '_'.join(train_folds),
            'Validation Fold': validate_fold,
            'Test Fold': test_fold,
            'Best Threshold': best_threshold,
            'Validation Best F1': best_f1,
            'Test Precision': test_results['Precision'],
            'Test Recall': test_results['Recall'],
            'Test F1': test_results['F1'],
            'Test MCC': test_results['MCC']
        }
        results_list.append(result)

    # After all combinations
    if final_thresholds:
        average_threshold = np.mean(final_thresholds)
        logging.info(f"Final Threshold (Average of {len(final_thresholds)} combinations): {average_threshold}")

        # Save final threshold
        threshold_file = os.path.join(output_dir, 'final_threshold.txt')
        if os.path.exists(threshold_file):
            logging.info(f"Final threshold file {threshold_file} already exists. Skipping saving.")
        else:
            try:
                with open(threshold_file, 'w') as f:
                    f.write(f"Final Threshold (Average): {average_threshold}\n")
                logging.info(f"Final threshold saved to {threshold_file}")
            except Exception as e:
                logging.error(f"Error saving final threshold to file {threshold_file}: {e}")

        # Save individual thresholds
        thresholds_csv = os.path.join(output_dir, 'individual_thresholds.csv')
        if os.path.exists(thresholds_csv):
            logging.info(f"Individual thresholds file {thresholds_csv} already exists. Skipping saving.")
        else:
            try:
                pd.DataFrame({'Threshold': final_thresholds}).to_csv(thresholds_csv, index=False)
                logging.info(f"Individual thresholds saved to {thresholds_csv}")
            except Exception as e:
                logging.error(f"Error saving individual thresholds to CSV file {thresholds_csv}: {e}")
    else:
        logging.error("No thresholds were determined. Please check the input data and logs for issues.")

    # Save results to CSV
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_csv = os.path.join(output_dir, 'results.csv')
        if os.path.exists(results_csv):
            logging.info(f"Results file {results_csv} already exists. Skipping saving.")
        else:
            try:
                results_df.to_csv(results_csv, index=False)
                logging.info(f"Results saved to {results_csv}")
            except Exception as e:
                logging.error(f"Error saving results to CSV file {results_csv}: {e}")
    else:
        logging.error("No results were collected. Please check the input data and logs for issues.")

    logging.info("Von Heijne analysis completed.")
