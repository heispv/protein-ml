import os
import logging
from Bio import SeqIO
import pandas as pd

def filter_tsv_by_fasta_ids(fasta_file_path, tsv_file_path, output_tsv_path):
    """
    Filters the .tsv file based on IDs in the .fasta file.
    
    Args:
        fasta_file_path (str): Path to the fasta file.
        tsv_file_path (str): Path to the .tsv file to filter.
        output_tsv_path (str): Path to save the filtered .tsv file.
    """
    # Extract IDs from fasta file
    ids = set()
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        ids.add(record.id)
    logging.info(f"Extracted {len(ids)} IDs from {fasta_file_path}")

    # Load the tsv file into a pandas DataFrame
    df = pd.read_csv(tsv_file_path, sep='\t')
    logging.info(f"Loaded {len(df)} entries from {tsv_file_path}")

    # Filter the DataFrame based on IDs
    filtered_df = df[df['primary_accession'].isin(ids)]
    logging.info(f"Filtered down to {len(filtered_df)} entries for {output_tsv_path}")

    # Save the filtered DataFrame to tsv
    filtered_df.to_csv(output_tsv_path, sep='\t', index=False)
    logging.info(f"Saved filtered tsv file to {output_tsv_path}")

def process_all_fasta_files(fetched_dir, split_dir):
    """
    Process all .fasta files in the split_dir, filter the corresponding .tsv files,
    and save the filtered .tsv files in the same directories as the .fasta files.

    Args:
        fetched_dir (str): Directory containing the fetched .tsv files.
        split_dir (str): Directory containing the splitted data (train/test and cross-validation folds).
    """
    # Paths to the fetched .tsv files
    pos_tsv_file = os.path.join(fetched_dir, 'pos_filtered_proteins.tsv')
    neg_tsv_file = os.path.join(fetched_dir, 'neg_filtered_proteins.tsv')

    # Check that the tsv files exist
    if not os.path.exists(pos_tsv_file):
        logging.error(f"File {pos_tsv_file} does not exist.")
        return
    if not os.path.exists(neg_tsv_file):
        logging.error(f"File {neg_tsv_file} does not exist.")
        return

    # For each .fasta file in split_dir
    for root, dirs, files in os.walk(split_dir):
        for file in files:
            if file.endswith('.fasta'):
                fasta_file_path = os.path.join(root, file)
                logging.info(f"Processing fasta file {fasta_file_path}")

                # Determine whether it's pos or neg based on the file path
                if 'pos' in root:
                    tsv_file_path = pos_tsv_file
                elif 'neg' in root:
                    tsv_file_path = neg_tsv_file
                else:
                    logging.warning(f"Cannot determine data type (pos/neg) from path {root}")
                    continue

                # Output tsv file path
                output_tsv_path = os.path.join(root, file.replace('.fasta', '.tsv'))
                
                if os.path.exists(output_tsv_path):
                    logging.info(f"Filtered TSV file {output_tsv_path} already exists. Skipping filtering for this FASTA file.")
                    print(f"Filtered TSV file {output_tsv_path} already exists. Skipping filtering for this FASTA file.")
                    continue
                else:
                    logging.info(f"Filtered TSV file {output_tsv_path} does not exist. Proceeding with filtering.")
                    print(f"Filtered TSV file {output_tsv_path} does not exist. Proceeding with filtering.")

                # Filter tsv
                filter_tsv_by_fasta_ids(fasta_file_path, tsv_file_path, output_tsv_path)
