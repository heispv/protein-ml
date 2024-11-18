# Protein Signal Peptide Prediction Pipeline

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Pipeline Overview](#pipeline-overview)
- [Data Processing](#data-processing)
- [Feature Extraction and Selection](#feature-extraction-and-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Benchmarking](#benchmarking)
- [Error Analysis](#error-analysis)
- [Performance Metrics](#performance-metrics)
- [Results](#results)
- [Future Directions](#future-directions)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

The **Protein Signal Peptide Prediction Pipeline** is a comprehensive bioinformatics tool designed to accurately predict signal peptides (SPs) in protein sequences. Signal peptides are short sequences at the N-terminus of proteins that direct them to the secretory pathway, playing a crucial role in protein localization and function. Accurate prediction of SPs is essential for applications in drug discovery, vaccine development, and recombinant protein production.

This pipeline implements and compares two primary approaches for SP prediction:

1. **Von Heijne’s Algorithm:** A foundational statistical method utilizing a Position-Specific Weight Matrix (PSWM) to predict cleavage sites based on residue frequency patterns.
2. **Support Vector Machines (SVMs):** A machine learning model that leverages multidimensional sequence features to enhance prediction accuracy and robustness.

By integrating traditional statistical methods with advanced machine learning techniques, this pipeline aims to provide superior performance in SP prediction tasks.

## Features

- **Data Fetching:** Retrieves protein data from UniProtKB/SwissProt based on defined positive and negative queries.
- **Data Filtering:** Applies stringent criteria to curate a high-quality dataset of experimentally validated SPs and non-SPs.
- **Clustering:** Utilizes MMseqs2 to reduce dataset redundancy, ensuring diverse and non-redundant sequences.
- **Data Splitting:** Divides data into training and benchmarking sets with stratified sampling to maintain class balance.
- **Feature Extraction:** Computes various biochemical and structural properties essential for SP prediction.
- **Feature Selection:** Employs Random Forests to identify the most informative features for model training.
- **Model Training:** Implements both von Heijne’s PSWM-based method and SVM classifiers with hyperparameter tuning.
- **Evaluation & Benchmarking:** Assesses model performance using metrics like Matthews Correlation Coefficient (MCC), Precision, Recall, and F1-score.
- **Error Analysis:** Identifies common misclassification pitfalls, such as transmembrane helices and atypical SPs.
- **Docker Support:** Containerizes the entire pipeline for seamless deployment and reproducibility.

## Architecture

The pipeline is modular, consisting of multiple interconnected scripts, each handling a specific stage of the data processing and model training workflow. The primary components include:

- **Data Fetching and Filtering**
- **Clustering and Data Splitting**
- **Feature Extraction and Selection**
- **Model Training and Evaluation**
- **Benchmarking and Error Analysis**

![Pipeline Architecture](docs/pipeline_architecture.png)

## Project Structure

The project directory is organized as follows:

```
.
├── Dockerfile
├── LICENCE
├── README.md
├── __init__.py
├── config.py
├── data
│   ├── cleavage_site_seqs
│   │   ├── test
│   │   │   └── cleavage_site_sequences_test.fasta
│   │   └── train
│   │       ├── 1
│   │       │   └── pos
│   │       │       └── cleavage_site_sequences_train_1_pos.fasta
│   │       ├── 2
│   │       │   └── pos
│   │       │       └── cleavage_site_sequences_train_2_pos.fasta
│   │       ├── 3
│   │       │   └── pos
│   │       │       └── cleavage_site_sequences_train_3_pos.fasta
│   │       ├── 4
│   │       │   └── pos
│   │       │       └── cleavage_site_sequences_train_4_pos.fasta
│   │       ├── 5
│   │       │   └── pos
│   │       │       └── cleavage_site_sequences_train_5_pos.fasta
│   │       └── cleavage_site_sequences_train.fasta
│   ├── clustered_data
│   │   ├── negative
│   │   │   ├── cluster_results_i30_c40_neg_all_seqs.fasta
│   │   │   ├── cluster_results_i30_c40_neg_cluster.tsv
│   │   │   └── cluster_results_i30_c40_neg_rep_seq.fasta
│   │   └── positive
│   │       ├── cluster_results_i30_c40_pos_all_seqs.fasta
│   │       ├── cluster_results_i30_c40_pos_cluster.tsv
│   │       └── cluster_results_i30_c40_pos_rep_seq.fasta
│   ├── features
│   │   ├── feature_means.csv
│   │   ├── feature_stds.csv
│   │   ├── norm_protein_features.csv
│   │   ├── protein_features.csv
│   │   ├── selected_features
│   │   │   ├── final_top_20_features.csv
│   │   │   ├── fold_1_feature_importances.csv
│   │   │   ├── fold_2_feature_importances.csv
│   │   │   ├── fold_3_feature_importances.csv
│   │   │   ├── fold_4_feature_importances.csv
│   │   │   └── fold_5_feature_importances.csv
│   │   └── testing
│   │       ├── test_norm_protein_features.csv
│   │       └── test_protein_features.csv
│   ├── fetched_data
│   │   ├── neg_filtered_proteins.fasta
│   │   ├── neg_filtered_proteins.tsv
│   │   ├── pos_filtered_proteins.fasta
│   │   └── pos_filtered_proteins.tsv
│   ├── pipeline_execution.log
│   ├── results
│   │   ├── final_svm_model.joblib
│   │   ├── svm_benchmark
│   │   │   ├── benchmark_metrics.csv
│   │   │   ├── confusion_matrix.png
│   │   │   ├── error_analysis
│   │   │   │   ├── aa_composition_comparison.png
│   │   │   │   ├── avg_alpha_propensity_boxplot_fn_vs_tp.png
│   │   │   │   ├── avg_hydrophobicity_boxplot_fn_vs_tp.png
│   │   │   │   ├── avg_transmembrane_propensity_boxplot_fn_vs_tp.png
│   │   │   │   ├── max_alpha_propensity_boxplot_fn_vs_tp.png
│   │   │   │   ├── max_charge_abundance_boxplot_fn_vs_tp.png
│   │   │   │   ├── max_hydrophobicity_boxplot_fn_vs_tp.png
│   │   │   │   ├── max_transmembrane_propensity_boxplot_fn_vs_tp.png
│   │   │   │   ├── pos_max_charge_abundance_boxplot_fn_vs_tp.png
│   │   │   │   └── sequence_length_distribution_combined.png
│   │   │   ├── false_negatives_ids.csv
│   │   │   ├── false_positives_ids.csv
│   │   │   ├── true_negatives_ids.csv
│   │   │   └── true_positives_ids.csv
│   │   ├── svm_initial_5fold_results.csv
│   │   └── svm_refined_hyperparameters.csv
│   ├── splited_data
│   │   ├── test
│   │   │   ├── neg
│   │   │   │   ├── cluster_results_i30_c40_neg_rep_seq_test.fasta
│   │   │   │   └── cluster_results_i30_c40_neg_rep_seq_test.tsv
│   │   │   └── pos
│   │   │       ├── cluster_results_i30_c40_pos_rep_seq_test.fasta
│   │   │       └── cluster_results_i30_c40_pos_rep_seq_test.tsv
│   │   └── train
│   │       ├── 1
│   │       │   ├── neg
│   │       │   │   ├── neg_fold_1.fasta
│   │       │   │   └── neg_fold_1.tsv
│   │       │   └── pos
│   │       │       ├── pos_fold_1.fasta
│   │       │       └── pos_fold_1.tsv
│   │       ├── 2
│   │       │   ├── neg
│   │       │   │   ├── neg_fold_2.fasta
│   │       │   │   └── neg_fold_2.tsv
│   │       │   └── pos
│   │       │       ├── pos_fold_2.fasta
│   │       │       └── pos_fold_2.tsv
│   │       ├── 3
│   │       │   ├── neg
│   │       │   │   ├── neg_fold_3.fasta
│   │       │   │   └── neg_fold_3.tsv
│   │       │   └── pos
│   │       │       ├── pos_fold_3.fasta
│   │       │       └── pos_fold_3.tsv
│   │       ├── 4
│   │       │   ├── neg
│   │       │   │   ├── neg_fold_4.fasta
│   │       │   │   └── neg_fold_4.tsv
│   │       │   └── pos
│   │       │       ├── pos_fold_4.fasta
│   │       │       └── pos_fold_4.tsv
│   │       ├── 5
│   │       │   ├── neg
│   │       │   │   ├── neg_fold_5.fasta
│   │       │   │   └── neg_fold_5.tsv
│   │       │   └── pos
│   │       │       ├── pos_fold_5.fasta
│   │       │       └── pos_fold_5.tsv
│   │       ├── neg
│   │       │   ├── cluster_results_i30_c40_neg_rep_seq_train.fasta
│   │       │   └── cluster_results_i30_c40_neg_rep_seq_train.tsv
│   │       └── pos
│   │           ├── cluster_results_i30_c40_pos_rep_seq_train.fasta
│   │           └── cluster_results_i30_c40_pos_rep_seq_train.tsv
│   ├── vonHeijne_results
│   │   ├── final_threshold.txt
│   │   ├── individual_thresholds.csv
│   │   ├── results.csv
│   │   ├── scoring_matrix_train_folds_1_2_3.csv
│   │   ├── scoring_matrix_train_folds_2_3_4.csv
│   │   ├── scoring_matrix_train_folds_3_4_5.csv
│   │   ├── scoring_matrix_train_folds_4_5_1.csv
│   │   └── scoring_matrix_train_folds_5_1_2.csv
│   └── vonHeijne_results_benchmark
│       ├── benchmark_results.csv
│       ├── detailed_results.csv
│       ├── scoring_matrix.csv
│       ├── von_heijne_fp_id.csv
│       └── von_heijne_tn_id.csv
├── data_classes.py
├── figures
│   ├── comparative_aa_composition
│   │   ├── aa_composition_comparison_test.png
│   │   └── aa_composition_comparison_train.png
│   ├── protein_length_dist
│   │   ├── protein_length_distribution_test.png
│   │   └── protein_length_distribution_train.png
│   ├── scientific_name
│   │   ├── scientific_name_classification_test.png
│   │   └── scientific_name_classification_train.png
│   ├── signal_peptide_length_dist
│   │   ├── sp_length_distribution_test.png
│   │   └── sp_length_distribution_train.png
│   ├── taxonomic_classification
│   │   ├── taxonomic_classification_test.png
│   │   └── taxonomic_classification_train.png
│   └── weblogo
│       ├── logo_test.png
│       └── logo_train.png
├── main.py
├── pipeline_01_data_fetcher.py
├── pipeline_01_data_filterer.py
├── pipeline_02_data_clusterer.py
├── pipeline_03_data_splitting.py
├── pipeline_04_cross_validation_split.py
├── pipeline_05_filter_tsv.py
├── pipeline_06_data_analysis.py
├── pipeline_07_vonHeijne_n_fold.py
├── pipeline_08_vonHeijne_benchmark.py
├── pipeline_09_svm_feature_collection.py
├── pipeline_10_feature_selection.py
├── pipeline_11_svm_hp_tuning.py
├── pipeline_12_svm_benchmark.py
├── pipeline_13_svm_error_analysis.py
├── project_tree.txt
├── requirements.txt
└── utils.py

56 directories, 126 files
```

## Installation

### Prerequisites

- **Docker:** Recommended for ease of installation and reproducibility. Download from [Docker Official Website](https://www.docker.com/get-started).
- **Python 3.8+** (if opting for manual installation)

### Using Docker (Recommended)

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/heispv/protein-ml.git
    cd protein-ml
    ```

2. **Build the Docker Image:**

    ```bash
    docker build -t protein-ml .
    ```

3. **Run the Docker Container:**

    ```bash
    docker run -it --rm -v $(pwd)/data:/app/data protein-ml
    ```

    *This command mounts the `data` directory from your local machine to the Docker container for data persistence.*

### Manual Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/heispv/protein-ml.git
    cd protein-ml
    ```

2. **Create a Virtual Environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Ensure MMseqs2 is Installed:**

    The pipeline utilizes MMseqs2 for clustering. Install it following the [MMseqs2 Installation Guide](https://github.com/soedinglab/MMseqs2/wiki/Installation).

## Configuration

All configuration parameters are defined in the `config.py` file. Key configurations include:

- **Data Directories:** Paths for fetched data, clustering results, splits, features, and results.
- **Queries:** Positive and negative queries for data fetching based on UniProtKB/SwissProt annotations.
- **Clustering Parameters:** MMseqs2 settings like identity and coverage thresholds.
- **Random Seed:** Ensures reproducibility across runs.

*Example `config.py`:*

```python
import os

# Base directory for data
DATA_DIR = os.path.join(os.getcwd(), 'data')

# Subdirectories
FETCHED_DIR = os.path.join(DATA_DIR, 'fetched_data')
CLUSTER_DIR = os.path.join(DATA_DIR, 'clustering_results')
SPLIT_DIR = os.path.join(DATA_DIR, 'splits')
FEATURES_DIR = os.path.join(DATA_DIR, 'features')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# Logging
LOG_FILE = os.path.join(DATA_DIR, 'pipeline.log')

# Queries
POSITIVE_QUERY = "taxonomy:Eukaryota AND reviewed:yes AND length:[40 TO *] AND existence:experimental AND signal:yes"
NEGATIVE_QUERY = "taxonomy:Eukaryota AND reviewed:yes AND length:[40 TO *] AND existence:experimental AND (location:Cytosol OR location:Nucleus OR location:Mitochondrion OR location:Plastid OR location:Peroxisome OR location:Cell_Membrane) AND signal:no"

# Clustering parameters
MMSEQS_IDENTITY = 0.30
MMSEQS_COVERAGE = 0.40
MMSEQS_FILE_PREFIX = 'cluster'

# Other parameters
BATCH_SIZE = 1000
NUM_FOLDS = 5
RANDOM_SEED = 42
```

*Modify `config.py` as per your requirements before running the pipeline.*

## Usage

The pipeline is orchestrated through the `main.py` script, which sequentially executes all pipeline stages. Ensure all configurations are set correctly in `config.py` before initiating the pipeline.

### Running the Pipeline

```bash
python main.py
```

*Alternatively, if using Docker, ensure the Docker container is running and execute the script within the container.*

## Pipeline Overview

The pipeline consists of the following sequential steps:

1. **Data Fetching (`pipeline_01_data_fetcher.py`):**
    - Retrieves protein data from UniProtKB/SwissProt based on predefined positive and negative queries.
    - Handles pagination and batch fetching to manage large datasets.

2. **Data Filtering (`pipeline_02_data_filterer.py`):**
    - Applies inclusion criteria to curate a high-quality dataset of experimentally validated SPs and non-SPs.
    - Ensures sequences are non-fragmented and meet length requirements.

3. **Clustering (`pipeline_03_data_clusterer.py`):**
    - Utilizes MMseqs2 to cluster protein sequences, reducing redundancy and computational load by enforcing a minimum sequence identity of 30% and coverage of 40%.

4. **Data Splitting (`pipeline_04_data_splitting.py` & `pipeline_05_cross_validation_split.py`):**
    - Splits the clustered data into training (80%) and benchmarking (20%) sets.
    - Employs stratified sampling to maintain class balance across splits.
    - Further divides the training set into cross-validation folds.

5. **Feature Extraction (`pipeline_06_feature_extraction.py`):**
    - Extracts biochemical and structural features from protein sequences, including amino acid composition, hydrophobicity, charge, alpha-helix propensity, and transmembrane propensity.

6. **Feature Selection (`pipeline_07_feature_selection.py`):**
    - Uses a Random Forest-based approach to assess feature importance.
    - Selects the top 20 most informative features for model training.

7. **Von Heijne’s Method Implementation (`pipeline_08_vonHeijne.py`):**
    - Implements the Position-Specific Weight Matrix (PSWM) approach to predict SP cleavage sites.
    - Optimizes threshold values through 5-fold cross-validation.

8. **Support Vector Machine Training (`pipeline_09_svm_training.py`):**
    - Trains an SVM classifier using the extracted and selected features.
    - Employs an RBF kernel with hyperparameters `C` and `γ` optimized via grid search and cross-validation.

9. **Evaluation & Benchmarking (`pipeline_10_evaluation.py`):**
    - Evaluates both von Heijne’s method and the SVM classifier using metrics such as MCC, Precision, Recall, and F1-score.
    - Compares performance to determine the superior approach.

10. **Error Analysis (`pipeline_11_error_analysis.py`):**
    - Analyzes misclassifications, particularly false positives arising from transmembrane helices and false negatives from atypical SPs.
    - Generates visualizations to identify patterns and areas for improvement.

## Data Processing

The pipeline processes protein data through several stages to ensure a high-quality dataset suitable for accurate SP prediction:

- **Fetching:** Retrieves protein sequences and associated metadata from UniProtKB/SwissProt based on specific queries.
- **Filtering:** Applies stringent inclusion criteria to curate experimentally validated SPs and non-SPs.
- **Clustering:** Reduces redundancy using MMseqs2, ensuring diverse and non-redundant sequences.
- **Splitting:** Divides data into training and benchmarking sets with stratified sampling to maintain class balance.

## Feature Extraction and Selection

Features are crucial for training the SVM classifier. The pipeline extracts various biochemical and structural properties, including:

1. **Amino Acid Composition:** Frequencies of 20 residues in the first 20 positions of the sequence, calculated as the proportion of each residue in this region.
2. **Hydrophobicity:** Maximal and average hydrophobicity values computed using the Kyte-Doolittle scale within a sliding window of size 5 over the first 40 residues.
3. **Charge:** Maximal abundance of positively charged residues (K, R) and the normalized position of this maximum, calculated using a sliding window of size 3 over the first 40 residues.
4. **Alpha-Helix Propensity:** Average and maximal alpha-helix propensities, computed using established scales in a sliding window of size 7 over the first 40 residues.
5. **Transmembrane Propensity:** Average and maximal transmembrane tendencies, calculated using a sliding window of size 7 over the first 40 residues.

**Feature Selection:**

- Utilizes a Random Forest-based approach to assess feature importance.
- Selects the top 20 most informative features based on their importance scores across cross-validation folds.
- Ensures that only the most predictive features are retained for model training, enhancing efficiency and performance.

## Model Training and Evaluation

### Von Heijne’s Method

- **Approach:** Utilizes a Position-Specific Weight Matrix (PSWM) to analyze residue frequencies in a sliding window of 15 positions along the N-terminal region.
- **Threshold Optimization:** Determines the optimal threshold for SP prediction through 5-fold cross-validation, averaging the best thresholds across all folds.

### Support Vector Machines (SVMs)

- **Approach:** Maps sequences into a multidimensional feature space and finds an optimal hyperplane to separate SPs from non-SPs using an RBF kernel.
- **Hyperparameter Tuning:** Optimizes the regularization parameter `C` and kernel coefficient `γ` via grid search and 5-fold cross-validation to maximize the Matthews Correlation Coefficient (MCC).
- **Training:** Fits the SVM model on the training data using the selected features.

### Evaluation Metrics

- **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications, considering true and false positives and negatives.
- **Precision:** The ratio of true positives to the sum of true and false positives.
- **Recall:** The ratio of true positives to the sum of true positives and false negatives.
- **F1-Score:** The harmonic mean of precision and recall.
- **False Positive Rate (FPR):** Specifically assesses the rate of transmembrane helices misclassified as SPs.

## Benchmarking

Benchmarking compares the performance of von Heijne’s method and the SVM classifier on a separate test dataset. The key findings include:

- **SVMs outperform von Heijne’s method** across all metrics:
  - **MCC:** 0.84 (SVM) vs. 0.68 (Von Heijne)
  - **Precision:** 0.85 (SVM) vs. 0.66 (Von Heijne)
  - **Recall:** 0.83 (SVM) vs. 0.77 (Von Heijne)
  - **F1-Score:** 0.84 (SVM) vs. 0.71 (Von Heijne)
- **False Positive Rates:**
  - **Transmembrane Helices FPR:** 0.21 (SVM) vs. 0.26 (Von Heijne)

These results demonstrate the superior accuracy and robustness of SVMs in distinguishing SPs from non-SPs, particularly in handling hydrophobic regions that mimic SP features.

## Error Analysis

Error analysis identified common misclassification pitfalls:

- **False Positives:** Predominantly transmembrane helices misclassified as SPs due to similar hydrophobic characteristics.
- **False Negatives:** SPs with atypical lengths or cleavage site compositions not captured effectively by the models.

Additional insights include:

- **Protein Length:** Longer proteins may have SPs located beyond the first 20 amino acids, leading to missed predictions when focusing solely on the N-terminal region.
- **Amino Acid Composition:** High abundance of leucine (L) in non-SPs complicates accurate prediction, as its presence closely resembles background distributions.

These findings highlight the need for refined models that account for sequence context and amino acid-specific patterns in greater detail.

## Performance Metrics

Performance of the models is evaluated using the following metrics:

1. **Matthews Correlation Coefficient (MCC):**

    \[
    MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}
    \]

2. **Precision:**

    \[
    Precision = \frac{TP}{TP + FP}
    \]

3. **Recall:**

    \[
    Recall = \frac{TP}{TP + FN}
    \]

4. **F1-Score:**

    \[
    F1\text{-}score = 2 \times \frac{Precision \times Recall}{Precision + Recall}
    \]

5. **False Positive Rate (FPR):**

    \[
    FPR = \frac{FP}{FP + TN}
    \]

These metrics provide a comprehensive assessment of model performance, balancing the trade-offs between sensitivity and specificity.

## Results

### Cross-Validation Results

| Method              | MCC  | Precision | Recall | F1-Score |
|---------------------|------|-----------|--------|----------|
| Von Heijne          | 0.68 | 0.66      | 0.77   | 0.71     |
| Support Vector M.   | 0.84 | 0.85      | 0.83   | 0.84     |

### Benchmarking Performance

| Method              | MCC  | Precision | Recall | F1-Score | FPR | TM Helix FPR |
|---------------------|------|-----------|--------|----------|-----|--------------|
| Von Heijne          | 0.65 | 0.64      | 0.76   | 0.69     | 0.05| 0.26         |
| Support Vector M.   | 0.82 | 0.75      | 0.94   | 0.84     | 0.04| 0.21         |

**Key Observations:**

- **SVMs significantly outperform** von Heijne’s method across all metrics.
- **Lower FPR for transmembrane helices** indicates better discrimination between SPs and transmembrane regions.
- **Protein length and amino acid composition** are critical factors influencing prediction accuracy.

## Future Directions

To further enhance SP prediction accuracy and address existing limitations, the following approaches are proposed:

1. **Advanced Sequence Representations:**
    - Replace one-hot encoding with embeddings derived from pre-trained models like ProtT5 or ESM to capture contextual and evolutionary information across entire protein sequences.

2. **Transformer-Based Models:**
    - Implement transformer architectures to leverage global sequence context and attention mechanisms, enabling the model to focus on critical regions regardless of their position within the sequence.

3. **Extended Feature Sets:**
    - Incorporate additional features such as evolutionary profiles, structural predictions, and post-translational modifications to enrich the feature space.

4. **Handling Longer Proteins:**
    - Develop models capable of processing full-length sequences to accurately predict SPs located beyond the initial N-terminal region.

5. **Resource Optimization:**
    - Optimize computational resources to facilitate the training and deployment of complex models in resource-limited environments.

These advancements hold the potential to overcome current challenges, providing more accurate and reliable SP predictions.

## Dependencies

All Python dependencies are listed in the `requirements.txt` file:

```
requests
pandas
numpy
matplotlib
seaborn
biopython
scikit-learn
joblib
mmseqs2
```

*Install dependencies using:*

```bash
pip install -r requirements.txt
```

**Note:** Ensure that external tools like MMseqs2 are installed and accessible in your system's PATH.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a Feature Branch**

    ```bash
    git checkout -b feature/YourFeature
    ```

3. **Commit Your Changes**

    ```bash
    git commit -m "Add some feature"
    ```

4. **Push to the Branch**

    ```bash
    git push origin feature/YourFeature
    ```

5. **Open a Pull Request**

Please ensure that your contributions adhere to the project’s coding standards and include appropriate tests and documentation.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please contact:

- **Name:** Peyman Vahidi
- **Email:** peymanvahidi1998@gmail.com
- **GitHub:** [heispv](https://github.com/heispv)

---

**Acknowledgments:** The author would like to express sincere gratitude to Professor Savojardo Castrense for their invaluable support and guidance throughout this research project.

**Supplementary Information:** All supplementary materials, including datasets, code, and additional figures, are available in our GitHub repository.