import os

DATA_DIR = "data/"

# Data fetching
FETCHED_DIR = os.path.join(DATA_DIR, "fetched_data/")
BATCH_SIZE = 500
BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
POSITIVE_QUERY = "((taxonomy_id:2759) AND (reviewed:true) AND (length:[40 TO *]) AND (ft_signal_exp:*) AND (fragment:false))"
NEGATIVE_QUERY = "((existence:1) AND (taxonomy_id:2759) AND (length:[40 TO *]) AND (reviewed:true) AND (fragment:false) AND ((cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0173) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0039)) NOT (ft_signal:*))"
LOG_FILE = os.path.join(DATA_DIR, "pipeline_execution.log")

# Data clustering
CLUSTER_DIR = os.path.join(DATA_DIR, "clustered_data/")
MMSEQS_IDENTITY = 0.3
MMSEQS_COVERAGE = 0.4
MMSEQS_FILE_PREFIX = f"cluster_results_i{int(100*MMSEQS_IDENTITY)}_c{int(100*MMSEQS_COVERAGE)}"

SPLIT_DIR = os.path.join(DATA_DIR, "splited_data/")

NUM_FOLDS = 5

FEATURES_DIR = os.path.join('data', 'features')
# Feature extraction parameters
K_AA_DP = 22  # Number of N-terminal residues for AA and DP
K_OTHERS = 40  # Number of N-terminal residues for other features

RANDOM_SEED = 42

SELECTED_FEATURES_DIR = os.path.join(FEATURES_DIR, 'selected_features')

PROTEIN_FEATURES_FILE = os.path.join(FEATURES_DIR, 'protein_features.csv')


N_JOBS = -1  # Number of parallel jobs for cross-validation (-1 uses all processors)

RESULTS_DIR = os.path.join(DATA_DIR, 'results')

NORM_PROTEIN_FEATURES_FILE = os.path.join(FEATURES_DIR, 'norm_protein_features.csv')


EXPECTED_ID_COLUMN = 'accession_id'

TEST_FEATURES_DIR = os.path.join(FEATURES_DIR, 'testing')
TEST_NORM_PROTEIN_FEATURES_FILE = os.path.join(TEST_FEATURES_DIR, 'test_norm_protein_features.csv')


# Paths for SVM benchmarking and error analysis
SVM_BENCHMARK_DIR = os.path.join(RESULTS_DIR, 'svm_benchmark')
FALSE_NEGATIVES_IDS_FILE = os.path.join(SVM_BENCHMARK_DIR, 'false_negatives_ids.csv')
TRUE_POSITIVES_IDS_FILE = os.path.join(SVM_BENCHMARK_DIR, 'true_positives_ids.csv')
ERROR_ANALYSIS_OUTPUT_DIR = os.path.join(SVM_BENCHMARK_DIR, 'error_analysis')

# Test features file (not normalized)
TEST_FEATURES_DIR = os.path.join(FEATURES_DIR, 'testing')
TEST_PROTEIN_FEATURES_FILE = os.path.join(TEST_FEATURES_DIR, 'test_protein_features.csv')  # Not normalized

# Paths for test FASTA files
POS_FASTA_FILE = os.path.join(SPLIT_DIR, 'test', 'pos', f'{MMSEQS_FILE_PREFIX}_pos_rep_seq_test.fasta')
NEG_FASTA_FILE = os.path.join(SPLIT_DIR, 'test', 'neg', f'{MMSEQS_FILE_PREFIX}_neg_rep_seq_test.fasta')