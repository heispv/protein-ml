# main.py

from urllib.parse import quote
from config import (BASE_URL, POSITIVE_QUERY, NEGATIVE_QUERY, BATCH_SIZE, LOG_FILE,
                    SPLIT_DIR, MMSEQS_FILE_PREFIX, NUM_FOLDS, FETCHED_DIR, CLUSTER_DIR)
from utils import setup_logging
from pipeline_01_data_fetcher import create_session
from pipeline_01_data_filterer import get_pos_dataset, get_neg_dataset
from pipeline_02_data_clusterer import run_mmseqs
from pipeline_03_data_splitting import split_fasta_sequences
from pipeline_04_cross_validation_split import perform_cross_validation_split
from pipeline_05_filter_tsv import process_all_fasta_files  # Import the function
import logging
import os

def main():
    setup_logging()
    logging.info("Starting protein data processing")
    session = create_session()
    
    pos_url = f"{BASE_URL}?format=json&query={quote(POSITIVE_QUERY)}&size={BATCH_SIZE}"
    neg_url = f"{BASE_URL}?format=json&query={quote(NEGATIVE_QUERY)}&size={BATCH_SIZE}"
    
    print("Processing positive dataset:")
    get_pos_dataset(pos_url, "pos_filtered_proteins", session)
    
    print("\nProcessing negative dataset:")
    get_neg_dataset(neg_url, "neg_filtered_proteins", session)

    print("\nClustering positive data:")
    run_mmseqs('pos')
    print("Positive data clustering done.")
    
    print("\nClustering negative data:")
    run_mmseqs('neg')
    print("Negative data clustering done.")

    # Data Splitting
    print("\nSplitting data into training and testing sets:")
    data_types = ['pos', 'neg']
    for data_type in data_types:
        if data_type == 'pos':
            input_dir = os.path.join(CLUSTER_DIR, 'positive')
            fasta_file = os.path.join(input_dir, f"{MMSEQS_FILE_PREFIX}_pos_rep_seq.fasta")
        else:
            input_dir = os.path.join(CLUSTER_DIR, 'negative')
            fasta_file = os.path.join(input_dir, f"{MMSEQS_FILE_PREFIX}_neg_rep_seq.fasta")

        # Check if fasta file exists
        if not os.path.exists(fasta_file):
            logging.error(f"Fasta file {fasta_file} does not exist.")
            continue

        # Output directories
        train_output_dir = os.path.join(SPLIT_DIR, 'train', data_type)
        test_output_dir = os.path.join(SPLIT_DIR, 'test', data_type)

        # Create output directories if they do not exist
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        # Output file paths
        base_filename = os.path.basename(fasta_file)
        train_output_path = os.path.join(train_output_dir, base_filename.replace('.fasta', '_train.fasta'))
        test_output_path = os.path.join(test_output_dir, base_filename.replace('.fasta', '_test.fasta'))

        # Split sequences
        split_fasta_sequences(fasta_file, train_output_path, test_output_path)
        print(f"Data splitting for {data_type} completed.")
        logging.info(f"Data splitting for {data_type} completed.")

    # Cross-Validation Splitting
    print("\nPerforming cross-validation splitting on training data:")
    train_data_dir = os.path.join(SPLIT_DIR, 'train')
    perform_cross_validation_split(train_data_dir, num_folds=NUM_FOLDS)
    print("Cross-validation data splitting completed.")

    # Filtering .tsv files
    print("\nFiltering .tsv files based on fasta IDs:")
    fetched_data_dir = FETCHED_DIR
    splitted_data_dir = SPLIT_DIR
    process_all_fasta_files(fetched_data_dir, splitted_data_dir)
    print("Filtering of .tsv files completed.")

    logging.info("Protein data processing completed")
    print(f"\nLog file saved as: {LOG_FILE}")

if __name__ == "__main__":
    main()
