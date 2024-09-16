import os
import random
import logging
from Bio import SeqIO

def split_fasta_sequences(fasta_file_path, train_output_path, test_output_path, train_ratio=0.8):
    """
    Splits sequences from a fasta file into training and testing sets.

    Args:
        fasta_file_path (str): Path to the input fasta file.
        train_output_path (str): Path to save the training fasta file.
        test_output_path (str): Path to save the testing fasta file.
        train_ratio (float): Proportion of data to be used for training (between 0 and 1).
    """
    # Read all sequences
    sequences = list(SeqIO.parse(fasta_file_path, 'fasta'))
    total_sequences = len(sequences)
    logging.info(f"Read {total_sequences} sequences from {fasta_file_path}")

    # Shuffle sequences
    random.shuffle(sequences)
    logging.info("Shuffled sequences")

    # Split sequences
    train_size = int(total_sequences * train_ratio)
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]
    logging.info(f"Split sequences into {len(train_sequences)} training and {len(test_sequences)} testing sequences")

    # Save train sequences
    SeqIO.write(train_sequences, train_output_path, 'fasta')
    logging.info(f"Saved training sequences to {train_output_path}")

    # Save test sequences
    SeqIO.write(test_sequences, test_output_path, 'fasta')
    logging.info(f"Saved testing sequences to {test_output_path}")
