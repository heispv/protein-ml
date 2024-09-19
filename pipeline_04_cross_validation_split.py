import os
import random
import logging
from Bio import SeqIO

def perform_cross_validation_split(train_data_dir, num_folds=5):
    """
    Splits the training data into specified number of folds for cross-validation.
    If the cross-validation split files already exist, the function will skip the splitting step.
    
    Args:
        train_data_dir (str): Path to the training data directory containing 'pos' and 'neg' subdirectories.
        num_folds (int): Number of folds to split the data into.
    """
    # Subdirectories for positive and negative data
    data_types = ['pos', 'neg']

    # Check if cross-validation split files already exist
    all_files_exist = True
    for fold_num in range(1, num_folds + 1):
        for data_type in data_types:
            output_dir = os.path.join(train_data_dir, str(fold_num), data_type)
            output_file = os.path.join(output_dir, f"{data_type}_fold_{fold_num}.fasta")
            if not os.path.exists(output_file):
                all_files_exist = False
                break
        if not all_files_exist:
            break

    if all_files_exist:
        message = f"Cross-validation split files already exist in {train_data_dir}."
        logging.info(message)
        logging.info("Skipping cross-validation splitting step...")
        print(message)
        print("Skipping cross-validation splitting step...")
        return

    # Sequences dictionary to hold sequences for 'pos' and 'neg'
    sequences = {'pos': [], 'neg': []}
    
    # Read sequences from 'pos' and 'neg' directories
    for data_type in data_types:
        data_path = os.path.join(train_data_dir, data_type)
        if not os.path.exists(data_path):
            logging.error(f"Directory {data_path} does not exist.")
            continue
        
        # Get all fasta files in the directory
        fasta_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.fasta')]
        for fasta_file in fasta_files:
            seqs = list(SeqIO.parse(fasta_file, 'fasta'))
            sequences[data_type].extend(seqs)
            logging.info(f"Read {len(seqs)} sequences from {fasta_file}")
    
    # Shuffle sequences
    for data_type in data_types:
        random.shuffle(sequences[data_type])
        logging.info(f"Shuffled {data_type} sequences")
    
    # Split sequences into folds
    folds = {str(i): {'pos': [], 'neg': []} for i in range(1, num_folds+1)}
    for data_type in data_types:
        fold_size = len(sequences[data_type]) // num_folds
        for i in range(num_folds):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < num_folds - 1 else len(sequences[data_type])
            folds[str(i+1)][data_type] = sequences[data_type][start_idx:end_idx]
            logging.info(f"Fold {i+1}: {len(folds[str(i+1)][data_type])} {data_type} sequences")
    
    # Create new directories and save sequences
    for fold_num, fold_data in folds.items():
        for data_type in data_types:
            output_dir = os.path.join(train_data_dir, fold_num, data_type)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{data_type}_fold_{fold_num}.fasta")
            SeqIO.write(fold_data[data_type], output_file, 'fasta')
            logging.info(f"Saved {len(fold_data[data_type])} {data_type} sequences to {output_file}")
    
    logging.info("Cross-validation data splitting completed")
