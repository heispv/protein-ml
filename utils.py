import os
import logging
from datetime import datetime
from config import DATA_DIR, LOG_FILE, POSITIVE_QUERY, NEGATIVE_QUERY, BATCH_SIZE

def setup_logging():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
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
    