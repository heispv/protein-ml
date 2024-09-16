import glob
import os
import logging

from utils import run_command
from config import FETCHED_DIR, CLUSTER_DIR, MMSEQS_FILE_PREFIX, MMSEQS_IDENTITY, MMSEQS_COVERAGE

logger = logging.getLogger(__name__)

def run_mmseqs(data_type: str) -> None:
    """
    Run MMseqs clustering on the specified data type.

    Args:
        data_type (str): Either 'pos' or 'neg' for positive or negative data.

    Raises:
        ValueError: If data_type is neither 'pos' nor 'neg'.
    """
    if data_type not in {'pos', 'neg'}:
        raise ValueError("data_type must be either 'pos' or 'neg'")

    proteins_file = f"{data_type}_filtered_proteins.fasta"
    output_dir = os.path.join(CLUSTER_DIR, "positive" if data_type == "pos" else "negative")
    
    mmseqs_command = [
        "mmseqs", "easy-cluster",
        os.path.join(FETCHED_DIR, proteins_file),
        f"{MMSEQS_FILE_PREFIX}_{data_type}",
        "tmp",
        "--min-seq-id", str(MMSEQS_IDENTITY),
        "-c", str(MMSEQS_COVERAGE),
        "--cov-mode", "0",
        "--cluster-mode", "1"
    ]

    try:
        run_command(mmseqs_command)
    except Exception as e:
        logger.error("Failed to execute MMseqs command: %s", e)
        raise

    # Remove the temporary directory
    try:
        run_command(['rm', '-r', 'tmp'])
        logger.info("Temporary files removed.")
    except Exception as e:
        logger.warning("Failed to remove temporary directory: %s", e)

    # Create the output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Directory %s created/confirmed.", output_dir)
    except Exception as e:
        logger.error("Failed to create directory %s: %s", output_dir, e)
        raise

    # Move the result files
    files_to_move = glob.glob(f'{MMSEQS_FILE_PREFIX}_{data_type}*')

    if files_to_move:
        try:
            run_command(['mv'] + files_to_move + [output_dir])
            logger.info("Moved files to %s: %s", output_dir, files_to_move)
        except Exception as e:
            logger.error("Failed to move files to %s: %s", output_dir, e)
            raise
    else:
        logger.warning("No files matching %s_%s* were found.", MMSEQS_FILE_PREFIX, data_type)

    logger.info("Data splitting for %s completed.", data_type)