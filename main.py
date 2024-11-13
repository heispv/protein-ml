# main.py

from urllib.parse import quote
from config import (
    BASE_URL,
    POSITIVE_QUERY,
    NEGATIVE_QUERY,
    BATCH_SIZE,
    LOG_FILE,
    DATA_DIR,
    SPLIT_DIR,
    MMSEQS_FILE_PREFIX,
    NUM_FOLDS,
    FETCHED_DIR,
    CLUSTER_DIR,
    MAX_LENGTH,
    PROTEIN_LENGTH_DIST_DIR,
    SIGNAL_PEPTIDE_LENGTH_DIST_DIR,
    COMPARATIVE_AA_COMPOSITION_DIR,
    TAXONOMIC_CLASSIFICATION_DIR,
    SCIENTIFIC_NAME_CLASSIFICATION_DIR,
    CLEAVAGE_SITE_SEQS_DIR,
    VON_HEIJNE_RESULTS_DIR,
    VON_HEIJNE_BENCHMARK_RESULTS_DIR,
)
from utils import setup_logging
from pipeline_01_data_fetcher import create_session
from pipeline_01_data_filterer import get_pos_dataset, get_neg_dataset
from pipeline_02_data_clusterer import run_mmseqs
from pipeline_03_data_splitting import split_fasta_sequences
from pipeline_04_cross_validation_split import perform_cross_validation_split
from pipeline_05_filter_tsv import process_all_fasta_files
from pipeline_06_data_analysis import (
    analyze_protein_length_distribution,
    analyze_signal_peptide_length_distribution,
    compare_amino_acid_composition,
    plot_taxonomic_classification,
    plot_scientific_name_classification,
    extract_cleavage_site_sequences,
)
from pipeline_07_vonHeijne_n_fold import perform_vonHeijne_analysis
from pipeline_08_vonHeijne_benchmark import perform_vonHeijne_benchmark_analysis
from pipeline_09_svm_feature_collection import svm_extract_features_pipeline
from pipeline_10_feature_selection import perform_feature_selection
from pipeline_11_svm_hp_tuning import perform_svm_hyperparameter_tuning
from pipeline_12_svm_benchmark import perform_svm_benchmark
from pipeline_13_svm_error_analysis import perform_svm_error_analysis
import logging
import os


def main():
    # Setup logging
    setup_logging()
    logging.info("Starting protein data processing")
    
    # Create a session for data fetching
    session = create_session()
    
    # Construct URLs for positive and negative datasets
    pos_url = f"{BASE_URL}?format=json&query={quote(POSITIVE_QUERY)}&size={BATCH_SIZE}"
    neg_url = f"{BASE_URL}?format=json&query={quote(NEGATIVE_QUERY)}&size={BATCH_SIZE}"
    
    # Fetch and filter positive dataset
    print("Processing positive dataset:")
    get_pos_dataset(pos_url, "pos_filtered_proteins", session)
    
    # Fetch and filter negative dataset
    print("\nProcessing negative dataset:")
    get_neg_dataset(neg_url, "neg_filtered_proteins", session)
    
    # Clustering datasets
    for data_type in ['pos', 'neg']:
        print(f"\nClustering {data_type} data:")
        run_mmseqs(data_type)
        print(f"{data_type.capitalize()} data clustering done.")
    
    # Data Splitting into training and testing sets
    print("\nSplitting data into training and testing sets:")
    data_types = ['pos', 'neg']
    for data_type in data_types:
        input_dir = os.path.join(CLUSTER_DIR, 'positive' if data_type == 'pos' else 'negative')
        fasta_file = os.path.join(input_dir, f"{MMSEQS_FILE_PREFIX}_{data_type}_rep_seq.fasta")
        
        # Check if fasta file exists
        if not os.path.exists(fasta_file):
            logging.error(f"Fasta file {fasta_file} does not exist.")
            continue
        
        # Define output directories
        train_output_dir = os.path.join(SPLIT_DIR, 'train', data_type)
        test_output_dir = os.path.join(SPLIT_DIR, 'test', data_type)
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)
        
        # Define output file paths
        base_filename = os.path.basename(fasta_file)
        train_output_path = os.path.join(train_output_dir, base_filename.replace('.fasta', '_train.fasta'))
        test_output_path = os.path.join(test_output_dir, base_filename.replace('.fasta', '_test.fasta'))
        
        # Split sequences
        split_fasta_sequences(fasta_file, train_output_path, test_output_path)
        print(f"Data splitting for {data_type} completed.")
        logging.info(f"Data splitting for {data_type} completed.")
    
    # Cross-Validation Splitting on Training Data
    print("\nPerforming cross-validation splitting on training data:")
    perform_cross_validation_split(
        train_data_dir=os.path.join(SPLIT_DIR, 'train'),
        num_folds=NUM_FOLDS
    )
    print("Cross-validation data splitting completed.")
    
    # Filtering .tsv Files Based on Fasta IDs
    print("\nFiltering .tsv files based on fasta IDs:")
    process_all_fasta_files(
        fetched_dir=FETCHED_DIR,
        split_dir=SPLIT_DIR
    )
    print("Filtering of .tsv files completed.")
    
    # Data Analysis: Protein Length Distribution
    print("\nAnalyzing protein length distribution:")
    analyze_protein_length_distribution(
        split_dir=SPLIT_DIR,
        output_dir=PROTEIN_LENGTH_DIST_DIR,
        max_length=MAX_LENGTH
    )
    print("Protein length distribution analysis completed.")
    
    # Data Analysis: Signal Peptide Length Distribution
    print("\nAnalyzing signal peptide length distribution:")
    analyze_signal_peptide_length_distribution(
        split_dir=SPLIT_DIR,
        output_dir=SIGNAL_PEPTIDE_LENGTH_DIST_DIR,
        max_length=None
    )
    print("Signal peptide length distribution analysis completed.")
    
    # Comparative Amino Acid Composition Analysis
    print("\nComparing amino acid composition of SP sequences:")
    compare_amino_acid_composition(
        split_dir=SPLIT_DIR,
        output_dir=COMPARATIVE_AA_COMPOSITION_DIR
    )
    print("Amino acid composition comparison completed.")
    
    # Taxonomic Classification Analysis
    print("\nAnalyzing taxonomic classification:")
    plot_taxonomic_classification(
        split_dir=SPLIT_DIR,
        output_dir=TAXONOMIC_CLASSIFICATION_DIR
    )
    print("Taxonomic classification analysis completed.")
    
    # Scientific Name Classification Analysis
    print("\nAnalyzing scientific name classification:")
    plot_scientific_name_classification(
        split_dir=SPLIT_DIR,
        output_dir=SCIENTIFIC_NAME_CLASSIFICATION_DIR,
        num_classifications=7
    )
    print("Scientific name classification analysis completed.")
    
    # Extraction of Cleavage Site Sequences
    print("\nExtracting cleavage site sequences:")
    extract_cleavage_site_sequences(
        split_dir=SPLIT_DIR,
        output_dir=CLEAVAGE_SITE_SEQS_DIR
    )
    print("Cleavage site sequence extraction completed.")
    
    # Von Heijne Analysis
    print("\nPerforming Von Heijne analysis:")
    perform_vonHeijne_analysis(
        cleavage_site_seqs_dir=os.path.join(CLEAVAGE_SITE_SEQS_DIR, 'train'),
        splitted_data_dir=os.path.join(DATA_DIR, 'splited_data'),
        output_dir=VON_HEIJNE_RESULTS_DIR
    )
    print("Von Heijne analysis completed.")
    
    # Von Heijne Benchmark Analysis
    print("\nPerforming Von Heijne benchmark analysis:")
    perform_vonHeijne_benchmark_analysis(
        cleavage_site_seqs_file=os.path.join(CLEAVAGE_SITE_SEQS_DIR, 'train', 'cleavage_site_sequences_train.fasta'),
        splitted_data_dir=os.path.join(DATA_DIR, 'splited_data', 'test'),
        threshold_file=os.path.join(VON_HEIJNE_RESULTS_DIR, 'final_threshold.txt'),
        output_dir=VON_HEIJNE_BENCHMARK_RESULTS_DIR
    )
    print("Von Heijne benchmark analysis completed.")
    
    # Final Logging and Output
    logging.info("Protein data processing completed")
    print(f"\nLog file saved as: {LOG_FILE}")
    
    # Feature Extraction Pipeline
    print("\nExtracting features from sequences:")
    svm_extract_features_pipeline()
    print("Feature extraction completed.")
    
    # Feature Selection
    print("\nPerforming feature selection:")
    perform_feature_selection()
    print("Feature selection completed.")
    
    # SVM Hyperparameter Tuning
    print("\nPerforming SVM hyperparameter tuning:")
    perform_svm_hyperparameter_tuning()
    print("SVM hyperparameter tuning completed.")
    
    # SVM Benchmarking
    print("\nPerforming SVM benchmarking:")
    perform_svm_benchmark()
    print("SVM benchmarking completed.")
    
    # SVM Error Analysis
    print("\nPerforming SVM error analysis:")
    perform_svm_error_analysis()
    print("SVM error analysis completed.")


if __name__ == "__main__":
    main()
