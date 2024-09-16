# Protein Data Processing Pipeline

## Overview

This project provides a pipeline for processing protein data from the UniProt database. It automates the fetching, filtering, clustering, splitting, and preparation of protein datasets for machine learning tasks, such as classification or sequence analysis.

The pipeline includes steps for:

- Fetching protein data from the UniProt REST API based on specified queries.
- Filtering the data to include only relevant proteins.
- Clustering the proteins using MMseqs2 to reduce redundancy.
- Splitting the data into training and testing sets.
- Performing cross-validation splitting.
- Filtering metadata files (`.tsv`) based on the sequences included in each split.

**Note:** The `data/` and `experiment/` directories are included in the `.gitignore` file and are not pushed to the remote repository. These directories, along with their contents, will be created when you run the pipeline.

## Repository

You can find the project repository on GitHub:

[https://github.com/heispv/protein-ml](https://github.com/heispv/protein-ml)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Steps](#pipeline-steps)
  - [1. Data Fetching](#1-data-fetching)
  - [2. Data Filtering](#2-data-filtering)
  - [3. Data Clustering](#3-data-clustering)
  - [4. Data Splitting](#4-data-splitting)
  - [5. Cross-Validation Splitting](#5-cross-validation-splitting)
  - [6. Filtering Metadata Files](#6-filtering-metadata-files)
- [Directory Structure](#directory-structure)
- [Logging](#logging)
- [Customization](#customization)
- [Error Handling](#error-handling)
- [Cleaning Up](#cleaning-up)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Python 3.6 or higher** is installed.
- **pip** package manager is available.
- The following Python packages are installed:
  - `requests`
  - `biopython`
  - `pandas`
- **MMseqs2** is installed and available in your system's PATH.
- Access to the internet to fetch data from the UniProt REST API.

### Installing Python Packages

You can install the required Python packages using the following command:

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with the following content:

```
requests
biopython
pandas
```

Alternatively, install them individually:

```bash
pip install requests biopython pandas
```

### Installing MMseqs2

MMseqs2 is a software suite for fast and sensitive protein sequence searching and clustering.

- Download MMseqs2 from the [official website](https://mmseqs.com/).
- Follow the installation instructions for your operating system.
- Ensure that the `mmseqs` command is accessible from the command line (i.e., MMseqs2 is in your PATH).

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/heispv/protein-ml.git
```

Navigate to the project directory:

```bash
cd protein-ml
```

Install the required Python packages as described in the [Prerequisites](#prerequisites) section.

## Project Structure

```
protein-ml/
├── config.py
├── data_classes.py
├── main.py
├── pipeline_01_data_fetcher.py
├── pipeline_01_data_filterer.py
├── pipeline_02_data_clusterer.py
├── pipeline_03_data_splitting.py
├── pipeline_04_cross_validation.py
├── pipeline_05_filter_tsv.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

- **config.py**: Contains configuration variables for the pipeline, such as queries, URLs, batch size, and directory paths.
- **data_classes.py**: Defines data structures for storing protein data.
- **main.py**: The main script that orchestrates the execution of the pipeline.
- **pipeline_01_data_fetcher.py**: Functions to fetch data from the UniProt REST API.
- **pipeline_01_data_filterer.py**: Filters the fetched data based on specified criteria.
- **pipeline_02_data_clusterer.py**: Runs MMseqs2 to cluster the protein sequences.
- **pipeline_03_data_splitting.py**: Splits the clustered data into training and testing sets.
- **pipeline_04_cross_validation.py**: Splits the training data further into folds for cross-validation.
- **pipeline_05_filter_tsv.py**: Filters the `.tsv` metadata files to include only sequences present in the `.fasta` files.
- **utils.py**: Utility functions for logging and running shell commands.
- **requirements.txt**: List of required Python packages.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: This file.

**Note:** The `data/` directory is not included in the repository and will be created when you run the pipeline.

## Configuration

All configuration variables are stored in `config.py`. Before running the pipeline, you may need to adjust the configuration to suit your needs.

Key configuration variables include:

- **DATA_DIR**: Base directory for data storage.
- **FETCHED_DIR**: Directory for fetched data.
- **CLUSTER_DIR**: Directory for clustered data.
- **SPLIT_DIR**: Directory for split data.
- **LOG_FILE**: Path to the log file.
- **BASE_URL**: Base URL for the UniProt REST API.
- **POSITIVE_QUERY**: Query string for fetching positive protein data.
- **NEGATIVE_QUERY**: Query string for fetching negative protein data.
- **BATCH_SIZE**: Number of records to fetch per API request.
- **MMSEQS_IDENTITY**: Minimum sequence identity for MMseqs2 clustering.
- **MMSEQS_COVERAGE**: Minimum sequence coverage for MMseqs2 clustering.
- **MMSEQS_FILE_PREFIX**: Prefix for MMseqs2 output files.
- **NUM_FOLDS**: Number of folds for cross-validation.

Ensure that the directory paths are correct and accessible.

## Usage

To run the pipeline, execute the `main.py` script:

```bash
python main.py
```

The script will perform the following steps:

1. **Data Fetching**: Fetch protein data from the UniProt REST API.
2. **Data Filtering**: Filter the fetched data based on specified criteria.
3. **Data Clustering**: Cluster the protein sequences using MMseqs2.
4. **Data Splitting**: Split the clustered data into training and testing sets.
5. **Cross-Validation Splitting**: Perform cross-validation splitting on the training data.
6. **Filtering Metadata Files**: Filter the `.tsv` metadata files to include only sequences present in the `.fasta` files.

**Note:** The `data/` directory and its subdirectories will be created automatically when you run the script.

## Pipeline Steps

### 1. Data Fetching

The pipeline fetches protein data from the UniProt REST API based on the queries defined in `config.py`. It uses batch requests to efficiently retrieve large datasets.

- **Positive Dataset**: Proteins that meet the criteria defined in `POSITIVE_QUERY`.
- **Negative Dataset**: Proteins that meet the criteria defined in `NEGATIVE_QUERY`.

Data is saved in `data/fetched_data/` as `.tsv` and `.fasta` files.

### 2. Data Filtering

The fetched data is filtered to include only relevant proteins:

- **Positive Proteins**: Proteins with signal peptides of at least 14 amino acids.
- **Negative Proteins**: Proteins with transmembrane helices.

Filtered data is saved in the same directory.

### 3. Data Clustering

The filtered sequences are clustered using MMseqs2 to reduce redundancy:

- Clusters sequences with at least `MMSEQS_IDENTITY` identity and `MMSEQS_COVERAGE` coverage.
- Outputs representative sequences.

Clustered data is saved in `data/clustered_data/positive/` and `data/clustered_data/negative/`.

### 4. Data Splitting

The clustered sequences are split into training and testing sets:

- **Training Set**: 80% of the data.
- **Testing Set**: 20% of the data.

Split data is saved in `data/splitted_data/train/` and `data/splitted_data/test/`.

### 5. Cross-Validation Splitting

The training data is further split into `NUM_FOLDS` folds for cross-validation:

- Each fold contains an equal portion of the training data.
- The original `train/pos/` and `train/neg/` directories are deleted after splitting.

Cross-validation data is saved in `data/splitted_data/train/{1,2,3,4,5}/`.

### 6. Filtering Metadata Files

The `.tsv` metadata files are filtered to include only the sequences present in the `.fasta` files:

- For each `.fasta` file in the test and cross-validation directories, a corresponding `.tsv` file is created.
- Filters the original `.tsv` files based on the `primary_accession` field matching the sequence IDs in the `.fasta` files.

Filtered `.tsv` files are saved in the same directories as the `.fasta` files.

## Directory Structure

After running the pipeline, the `data/` directory will have the following structure:

```
data/
├── fetched_data/
│   ├── pos_filtered_proteins.tsv
│   ├── pos_filtered_proteins.fasta
│   ├── neg_filtered_proteins.tsv
│   └── neg_filtered_proteins.fasta
├── clustered_data/
│   ├── positive/
│   │   └── cluster_results_i30_c40_pos_rep_seq.fasta
│   └── negative/
│       └── cluster_results_i30_c40_neg_rep_seq.fasta
├── splitted_data/
│   ├── train/
│   │   ├── 1/
│   │   │   ├── pos/
│   │   │   │   ├── pos_fold_1.fasta
│   │   │   │   └── pos_fold_1.tsv
│   │   │   └── neg/
│   │   │       ├── neg_fold_1.fasta
│   │   │       └── neg_fold_1.tsv
│   │   ├── 2/
│   │   │   └── ...
│   │   ├── 3/
│   │   ├── 4/
│   │   └── 5/
│   └── test/
│       ├── pos/
│       │   ├── pos_test.fasta
│       │   └── pos_test.tsv
│       └── neg/
│           ├── neg_test.fasta
│           └── neg_test.tsv
└── pipeline_execution.log
```

**Note:** This directory structure will be created when you run the pipeline. Since the `data/` directory is in the `.gitignore` file, it is not included in the repository.

## Logging

The pipeline's execution is logged to `data/pipeline_execution.log`. This log file contains detailed information about each step, including:

- Start and completion times.
- Number of records processed.
- Any errors or warnings encountered.

## Customization

You can customize the pipeline by modifying the configuration variables in `config.py`:

- **Queries**: Adjust `POSITIVE_QUERY` and `NEGATIVE_QUERY` to change the criteria for fetching proteins.
- **Batch Size**: Modify `BATCH_SIZE` to control the number of records fetched per API request.
- **Clustering Parameters**: Change `MMSEQS_IDENTITY` and `MMSEQS_COVERAGE` to adjust clustering sensitivity.
- **Cross-Validation Folds**: Set `NUM_FOLDS` to change the number of folds in cross-validation.
- **Data Directories**: Update `DATA_DIR`, `FETCHED_DIR`, `CLUSTER_DIR`, and `SPLIT_DIR` to change where data is stored.

## Error Handling

- Ensure that all required dependencies are installed and accessible.
- The pipeline checks for the existence of files and directories before proceeding.
- Errors and warnings are logged to `pipeline_execution.log`.
- If a step fails, the pipeline logs the error and continues to the next step where appropriate.

## Cleaning Up

To rerun the pipeline from scratch:

1. Delete the `data/` directory or its contents:

   ```bash
   rm -r data/
   ```

2. Ensure that the `pipeline_execution.log` file is also deleted if you want a fresh log.
3. Rerun the `main.py` script:

   ```bash
   python main.py
   ```

Alternatively, you can modify the pipeline to overwrite existing files or to check for existing data before processing.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.

   ```bash
   git clone https://github.com/yourusername/protein-ml.git
   ```

2. Create a new branch:

   ```bash
   git checkout -b feature/my-feature
   ```

3. Make your changes.
4. Commit your changes:

   ```bash
   git commit -am 'Add new feature'
   ```

5. Push to the branch:

   ```bash
   git push origin feature/my-feature
   ```

6. Create a new Pull Request on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

If you have any questions or need further assistance, please feel free to contact the project maintainers.

**Contact Information:**

- **Name:** Peyman Vahidi
- **Email:** [peyman.vahidi@studio.unibo.it](mailto:peyman.vahidi@studio.unibo.it)

*Replace `[peyman.vahidi@studio.unibo.it]` with your actual email address if you wish to include it.*

---

**Acknowledgments:**

- This project was developed as part of a course assignment.
- Special thanks to the course instructors and peers for their support.