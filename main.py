from urllib.parse import quote
from config import BASE_URL, POSITIVE_QUERY, NEGATIVE_QUERY, BATCH_SIZE, LOG_FILE
from utils import setup_logging
from data_fetcher import create_session
from data_processor import get_pos_dataset, get_neg_dataset
import logging

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

    logging.info("Protein data processing completed")
    print(f"\nLog file saved as: {LOG_FILE}")

if __name__ == "__main__":
    main()