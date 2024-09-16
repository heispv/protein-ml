import os
import logging
from datetime import datetime
from config import DATA_DIR, FETCHED_DIR, CLUSTER_DIR, LOG_FILE, POSITIVE_QUERY, NEGATIVE_QUERY, BATCH_SIZE, SPLIT_DIR
import shlex
import subprocess


def setup_logging():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(FETCHED_DIR):
        os.makedirs(FETCHED_DIR)
    if not os.path.exists(CLUSTER_DIR):
        os.makedirs(CLUSTER_DIR)
    if not os.path.exists(SPLIT_DIR):
        os.makedirs(SPLIT_DIR)
        
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        os.makedirs(FETCHED_DIR)
        os.makedirs(CLUSTER_DIR)
    
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logging.info(f"Logging initialized. Data will be saved in {DATA_DIR}")
    logging.info(f"Script started at {datetime.now()}")
    logging.info(f"Positive query: {POSITIVE_QUERY}")
    logging.info(f"Negative query: {NEGATIVE_QUERY}")
    logging.info(f"Batch size: {BATCH_SIZE}")

def run_command(command):
    """
    Run a shell command and capture its output.

    Parameters:
    command (str or list): The shell command to execute. It can be a string or a list of arguments.

    Returns:
    str: The standard output of the command.

    Raises:
    subprocess.CalledProcessError: If the command exits with a non-zero status.
    FileNotFoundError: If the executable in the command is not found.
    """
    # If command is a string, split it; if it's already a list, use it directly
    args = shlex.split(command) if isinstance(command, str) else command
    
    try:
        # Run the command and capture output
        result = subprocess.run(args, check=True, capture_output=True, text=False)
        
        # Log the standard output
        logging.info("Command executed successfully.")
        logging.debug("Command output: %s", result.stdout)
        
        return result.stdout
    
    except subprocess.CalledProcessError as e:
        logging.error("Command failed with exit code %s", e.returncode)
        logging.error("Error output: %s", e.stderr)
        raise
    except FileNotFoundError:
        logging.critical("The command '%s' was not found. Ensure it is installed and in the system PATH.", args[0])
        raise