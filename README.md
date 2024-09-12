# Protein Data Processor

This package is designed to fetch and process protein data from the UniProt database. It retrieves both positive and negative datasets based on specific criteria, processes the data, and saves it in TSV and FASTA formats.

## Features

- Fetches protein data from UniProt REST API
- Processes both positive and negative datasets
- Extracts relevant features from protein data
- Saves processed data in TSV and FASTA formats
- Implements logging for better tracking and debugging

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/protein-data-processor.git
   cd protein-data-processor
   ```

2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the protein data processor:

```
python -m protein_data_processor.main
```

This will start the data fetching and processing. The script will:
1. Fetch positive and negative protein datasets from UniProt
2. Process the data and extract relevant features
3. Save the processed data in TSV and FASTA formats
4. Generate a log file with processing details

## Configuration

You can modify the following parameters in `protein_data_processor/config.py`:

- `BATCH_SIZE`: Number of entries to fetch per API call
- `BASE_URL`: Base URL for the UniProt REST API
- `POSITIVE_QUERY`: Query string for positive dataset
- `NEGATIVE_QUERY`: Query string for negative dataset
- `DATA_DIR`: Directory to save output files

## Output

The script generates the following output in the `fetched_data` directory:

1. `pos_filtered_proteins.tsv`: TSV file containing processed positive protein data
2. `pos_filtered_proteins.fasta`: FASTA file containing sequences for positive proteins
3. `neg_filtered_proteins.tsv`: TSV file containing processed negative protein data
4. `neg_filtered_proteins.fasta`: FASTA file containing sequences for negative proteins
5. `protein_data_processing.log`: Log file with processing details

## Project Structure

```
protein_data_processor/
│
├── __init__.py
├── main.py
├── config.py
├── data_fetcher.py
├── data_processor.py
├── utils.py
└── models.py
```

- `main.py`: Entry point of the script
- `config.py`: Configuration parameters
- `data_fetcher.py`: Functions for fetching data from the API
- `data_processor.py`: Functions for processing protein data
- `utils.py`: Utility functions (e.g., logging setup)
- `models.py`: Data models used in the package