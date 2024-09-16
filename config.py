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
