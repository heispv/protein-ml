import glob
import os
import logging

from utils import run_command
from config import FETCHED_DIR, CLUSTER_DIR, MMSEQS_FILE_PREFIX, MMSEQS_IDENTITY, MMSEQS_COVERAGE

# Constants
PROTEINS_FILE = "pos_filtered_proteins.fasta"

# Construct MMseqs command as a list of arguments
MMSEQS_COMMAND = [
    "mmseqs", "easy-cluster",
    os.path.join(FETCHED_DIR, PROTEINS_FILE),
    MMSEQS_FILE_PREFIX,
    "tmp",
    "--min-seq-id", str(MMSEQS_IDENTITY),
    "-c", str(MMSEQS_COVERAGE),
    "--cov-mode", "0",
    "--cluster-mode", "1"
]

def mmseqs():
    try:
        # Run the MMseqs command
        output = run_command(MMSEQS_COMMAND)
        print("Command output:", output)
    except Exception as e:
        print(f"Failed to execute MMseqs command: {e}")
        logging.error(f"MMseqs command failed: {e}")

    # Remove the tmp directory
    run_command(['rm', '-r', 'tmp'])
    print("Temporary files removed.")

    # Create the CLUSTER_DIR directory
    run_command(['mkdir', '-p', CLUSTER_DIR])
    print(f"Directory {CLUSTER_DIR} created.")

    # Move the result files to CLUSTER_DIR
    files_to_move = glob.glob(f'{MMSEQS_FILE_PREFIX}*')

    if files_to_move:
        # Move the result files to CLUSTER_DIR
        run_command(['mv'] + files_to_move + [CLUSTER_DIR])
        print(f"Moved files to {CLUSTER_DIR}: {files_to_move}")
    else:
        logging.error(f"No files matching {MMSEQS_FILE_PREFIX}* were found.")
        print(f"No files found to move.")

    print("Data splitting completed.")
