import os

BATCH_SIZE = 500
BASE_URL = "https://rest.uniprot.org/uniprotkb/search"
POSITIVE_QUERY = "((taxonomy_id:2759) AND (reviewed:true) AND (length:[40 TO *]) AND (ft_signal_exp:*))"
NEGATIVE_QUERY = "((taxonomy_id:2759) AND (length:[40 TO *]) NOT (ft_signal:*) AND (existence:1) AND (reviewed:true) AND ((cc_scl_term:SL-0091) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0173) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0039)))"
DATA_DIR = "fetched_data"
LOG_FILE = os.path.join(DATA_DIR, "protein_data_processing.log")
