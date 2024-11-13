import os

# =============================================================================
# General Configuration
# =============================================================================
DATA_DIR = "data/"
RANDOM_SEED = 42


# =============================================================================
# Logging Configuration
# =============================================================================
LOG_FILE = os.path.join(DATA_DIR, "pipeline_execution.log")


# =============================================================================
# Data Fetching Configuration
# =============================================================================
FETCHED_DIR = os.path.join(DATA_DIR, "fetched_data/")
BATCH_SIZE = 500
BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
POSITIVE_QUERY = "((taxonomy_id:2759) AND (reviewed:true) AND (length:[40 TO *]) AND (ft_signal_exp:*) AND (fragment:false))"
NEGATIVE_QUERY = "((existence:1) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (reviewed:true) AND (fragment:false) AND ((cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0173) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0039)) NOT (ft_signal:*))"


# =============================================================================
# Data Clustering Configuration
# =============================================================================
CLUSTER_DIR = os.path.join(DATA_DIR, "clustered_data/")
MMSEQS_IDENTITY = 0.3
MMSEQS_COVERAGE = 0.4
MMSEQS_FILE_PREFIX = f"cluster_results_i{int(100 * MMSEQS_IDENTITY)}_c{int(100 * MMSEQS_COVERAGE)}"


# =============================================================================
# Data Splitting Configuration
# =============================================================================
SPLIT_DIR = os.path.join(DATA_DIR, "splited_data/")
NUM_FOLDS = 5


# =============================================================================
# Feature Extraction Configuration
# =============================================================================
FEATURES_DIR = os.path.join(DATA_DIR, "features")
SELECTED_FEATURES_DIR = os.path.join(FEATURES_DIR, 'selected_features')
PROTEIN_FEATURES_FILE = os.path.join(FEATURES_DIR, 'protein_features.csv')
NORM_PROTEIN_FEATURES_FILE = os.path.join(FEATURES_DIR, 'norm_protein_features.csv')


# =============================================================================
# Testing Configuration
# =============================================================================
TEST_FEATURES_DIR = os.path.join(FEATURES_DIR, 'testing')
TEST_NORM_PROTEIN_FEATURES_FILE = os.path.join(TEST_FEATURES_DIR, 'test_norm_protein_features.csv')
TEST_PROTEIN_FEATURES_FILE = os.path.join(TEST_FEATURES_DIR, 'test_protein_features.csv')  # Not normalized

# Paths for Test FASTA Files
POS_FASTA_FILE = os.path.join(SPLIT_DIR, 'test', 'pos', f'{MMSEQS_FILE_PREFIX}_pos_rep_seq_test.fasta')
NEG_FASTA_FILE = os.path.join(SPLIT_DIR, 'test', 'neg', f'{MMSEQS_FILE_PREFIX}_neg_rep_seq_test.fasta')


# =============================================================================
# SVM Configuration
# =============================================================================
N_JOBS = -1  # Number of parallel jobs for cross-validation (-1 uses all processors)

# Paths for SVM Benchmarking and Error Analysis
RESULTS_DIR = os.path.join(DATA_DIR, 'results')
SVM_BENCHMARK_DIR = os.path.join(RESULTS_DIR, 'svm_benchmark')
FALSE_NEGATIVES_IDS_FILE = os.path.join(SVM_BENCHMARK_DIR, 'false_negatives_ids.csv')
TRUE_POSITIVES_IDS_FILE = os.path.join(SVM_BENCHMARK_DIR, 'true_positives_ids.csv')
ERROR_ANALYSIS_OUTPUT_DIR = os.path.join(SVM_BENCHMARK_DIR, 'error_analysis')


# =============================================================================
# Analysis Configuration
# =============================================================================
MAX_LENGTH = 5000  # Max protein length for the figures


# =============================================================================
# Output Directories for Analyses
# =============================================================================
FIGURES_DIR =  "figures/"
PROTEIN_LENGTH_DIST_DIR = os.path.join(FIGURES_DIR, 'protein_length_dist')
SIGNAL_PEPTIDE_LENGTH_DIST_DIR = os.path.join(FIGURES_DIR, 'signal_peptide_length_dist')
COMPARATIVE_AA_COMPOSITION_DIR = os.path.join(FIGURES_DIR, 'comparative_aa_composition')
TAXONOMIC_CLASSIFICATION_DIR = os.path.join(FIGURES_DIR, 'taxonomic_classification')
SCIENTIFIC_NAME_CLASSIFICATION_DIR = os.path.join(FIGURES_DIR, 'scientific_name')

SCIENTIFIC_NAME_CLASS_NUM = 7 # Number of classification for the Scientific name


# =============================================================================
# Cleavage Site Sequences Configuration
# =============================================================================
CLEAVAGE_SITE_SEQS_DIR = os.path.join(DATA_DIR, 'cleavage_site_seqs')


# =============================================================================
# Von Heijne Analysis Configuration
# =============================================================================
VON_HEIJNE_RESULTS_DIR = os.path.join(DATA_DIR, 'vonHeijne_results')
VON_HEIJNE_BENCHMARK_RESULTS_DIR = os.path.join(DATA_DIR, 'vonHeijne_results_benchmark')


# Expected ID Column for Data Validation
EXPECTED_ID_COLUMN = 'accession_id'
