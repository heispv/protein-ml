# pipeline_13_svm_error_analysis.py

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from collections import Counter
import numpy as np
from config import (
    EXPECTED_ID_COLUMN,
    SVM_BENCHMARK_DIR,
    TEST_PROTEIN_FEATURES_FILE,
    POS_FASTA_FILE,
    NEG_FASTA_FILE,
    ERROR_ANALYSIS_OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

def perform_svm_error_analysis():
    """
    Performs error analysis on the SVM model's predictions by comparing
    false negatives and true positives.
    """
    logger.info("Starting SVM error analysis.")

    # Define the list of output files to check
    output_dir = ERROR_ANALYSIS_OUTPUT_DIR
    features = [
        'max_transmembrane_propensity',
        'avg_hydrophobicity',
        'avg_transmembrane_propensity',
        'max_hydrophobicity',
        'pos_max_charge_abundance',
        'avg_alpha_propensity',
        'max_alpha_propensity',
        'max_charge_abundance'
    ]
    output_files = [
        os.path.join(output_dir, 'aa_composition_fn_vs_tp.png'),
        os.path.join(output_dir, 'sequence_length_distribution_combined.png')
    ] + [os.path.join(output_dir, f'{feature}_boxplot_fn_vs_tp.png') for feature in features]
    
    # Check if all output files exist
    if all(os.path.exists(file) for file in output_files):
        logger.info("All error analysis output files already exist. Skipping SVM error analysis.")
        print("All error analysis output files already exist. Skipping SVM error analysis.")
        return
    else:
        logger.info("One or more error analysis output files are missing. Proceeding with SVM error analysis.")
        print("One or more error analysis output files are missing. Proceeding with SVM error analysis.")

    # Define paths
    fn_ids_file = os.path.join(SVM_BENCHMARK_DIR, 'false_negatives_ids.csv')
    tp_ids_file = os.path.join(SVM_BENCHMARK_DIR, 'true_positives_ids.csv')
    test_features_file = TEST_PROTEIN_FEATURES_FILE  # Not normalized
    pos_fasta_file = POS_FASTA_FILE
    neg_fasta_file = NEG_FASTA_FILE
    os.makedirs(output_dir, exist_ok=True)

    # Read FN and TP IDs
    try:
        fn_ids_df = pd.read_csv(fn_ids_file)
        tp_ids_df = pd.read_csv(tp_ids_file)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}")
        return
    except Exception as e:
        logger.error(f"Error reading FN/TP IDs: {e}")
        return

    id_column = EXPECTED_ID_COLUMN  # Ensure this matches your CSV files
    if id_column not in fn_ids_df.columns or id_column not in tp_ids_df.columns:
        available_fn_cols = ', '.join(fn_ids_df.columns)
        available_tp_cols = ', '.join(tp_ids_df.columns)
        logger.error(
            f"Expected column '{id_column}' not found.\n"
            f"Available columns in FN IDs: {available_fn_cols}\n"
            f"Available columns in TP IDs: {available_tp_cols}"
        )
        return

    fn_ids = fn_ids_df[id_column].dropna().astype(str).tolist()
    tp_ids = tp_ids_df[id_column].dropna().astype(str).tolist()

    logger.info(f"Number of False Negatives (FN): {len(fn_ids)}")
    logger.info(f"Number of True Positives (TP): {len(tp_ids)}")

    # Read test protein features (not normalized)
    try:
        test_features_df = pd.read_csv(test_features_file)
    except FileNotFoundError:
        logger.error(f"Test features file not found: {test_features_file}")
        return
    except Exception as e:
        logger.error(f"Error reading test features: {e}")
        return

    if id_column not in test_features_df.columns:
        available_columns = ', '.join(test_features_df.columns)
        logger.error(
            f"Expected column '{id_column}' not found in test_protein_features.csv.\n"
            f"Available columns: {available_columns}"
        )
        return

    # Read sequences from fasta files
    try:
        pos_sequences = SeqIO.to_dict(SeqIO.parse(pos_fasta_file, 'fasta'))
        neg_sequences = SeqIO.to_dict(SeqIO.parse(neg_fasta_file, 'fasta'))
    except FileNotFoundError as e:
        logger.error(f"FASTA file not found: {e.filename}")
        return
    except Exception as e:
        logger.error(f"Error reading FASTA files: {e}")
        return

    # Combine sequences
    all_sequences = {**pos_sequences, **neg_sequences}

    # Get sequences for FN and TP proteins
    fn_sequences = []
    missing_fn_ids = []
    for acc in fn_ids:
        if acc in all_sequences:
            fn_sequences.append(all_sequences[acc])
        else:
            missing_fn_ids.append(acc)

    tp_sequences = []
    missing_tp_ids = []
    for acc in tp_ids:
        if acc in all_sequences:
            tp_sequences.append(all_sequences[acc])
        else:
            missing_tp_ids.append(acc)

    if missing_fn_ids:
        logger.warning(f"{len(missing_fn_ids)} FN accession IDs are missing in sequences.")
    if missing_tp_ids:
        logger.warning(f"{len(missing_tp_ids)} TP accession IDs are missing in sequences.")

    # Function to get first 22 residues
    def get_first_22_residues(sequences):
        seqs_22 = []
        for seq_record in sequences:
            seq_str = str(seq_record.seq)
            if len(seq_str) >= 22:
                seqs_22.append(seq_str[:22])
            else:
                seqs_22.append(seq_str)  # Include shorter sequences as they are
        return seqs_22

    fn_seqs_22 = get_first_22_residues(fn_sequences)
    tp_seqs_22 = get_first_22_residues(tp_sequences)

    # Compute amino acid composition
    def compute_aa_composition(sequences):
        aa_counts = Counter()
        total_length = 0
        for seq in sequences:
            aa_counts.update(seq)
            total_length += len(seq)
        if total_length == 0:
            return {}
        aa_composition = {aa: (count / total_length * 100) for aa, count in aa_counts.items()}
        return aa_composition

    fn_aa_comp = compute_aa_composition(fn_seqs_22)
    tp_aa_comp = compute_aa_composition(tp_seqs_22)

    # Fill missing amino acids
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')  # Standard amino acids

    def fill_missing_aa(comp_dict, amino_acids):
        for aa in amino_acids:
            comp_dict.setdefault(aa, 0.0)
        return comp_dict

    fn_aa_comp = fill_missing_aa(fn_aa_comp, amino_acids)
    tp_aa_comp = fill_missing_aa(tp_aa_comp, amino_acids)

    # Create DataFrame for plotting
    data = []
    for aa in amino_acids:
        data.append({'Amino Acid': aa, 'Percentage': fn_aa_comp.get(aa, 0.0), 'Group': 'False Negatives'})
        data.append({'Amino Acid': aa, 'Percentage': tp_aa_comp.get(aa, 0.0), 'Group': 'True Positives'})
    aa_comp_df = pd.DataFrame(data)

    if aa_comp_df.empty:
        logger.warning("Amino acid composition DataFrame is empty. Skipping amino acid composition plot.")
    else:
        # Ensure that 'Group' has the correct categories
        aa_comp_df['Group'] = aa_comp_df['Group'].astype('category')

        # Plot bar plot
        sns.set(style='whitegrid')
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Amino Acid', y='Percentage', hue='Group', data=aa_comp_df, order=amino_acids)
        plt.title('Amino Acid Composition of First 22 Residues: FN vs TP')
        plt.xlabel('Amino Acid')
        plt.ylabel('Percentage')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'aa_composition_fn_vs_tp.png')
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved amino acid composition plot to {output_file}")

    # Task 2: Compare sequence lengths
    fn_lengths = [len(str(seq_record.seq)) for seq_record in fn_sequences]
    tp_lengths = [len(str(seq_record.seq)) for seq_record in tp_sequences]

    if not fn_lengths or not tp_lengths:
        logger.warning("One of the groups (FN or TP) has no sequences. Skipping sequence length plots.")
    else:
        length_data = pd.DataFrame({
            'Sequence Length': fn_lengths + tp_lengths,
            'Group': ['False Negatives'] * len(fn_lengths) + ['True Positives'] * len(tp_lengths)
        })

        # Ensure 'Group' is categorical
        length_data['Group'] = length_data['Group'].astype('category')

        # Plot histogram and density plot
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(
            data=length_data,
            x='Sequence Length',
            hue='Group',
            bins=30,
            kde=True,
            stat='density',
            common_norm=False
        )
        plt.title('Protein Sequence Length Distribution: FN vs TP')
        plt.xlabel('Sequence Length')
        plt.ylabel('Density')
        plt.tight_layout()
        output_file = os.path.join(output_dir, 'sequence_length_distribution_combined.png')
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved combined sequence length plot to {output_file}")

    # Task 3: Compare features
    features = [
        'max_transmembrane_propensity',
        'avg_hydrophobicity',
        'avg_transmembrane_propensity',
        'max_hydrophobicity',
        'pos_max_charge_abundance',
        'avg_alpha_propensity',
        'max_alpha_propensity',
        'max_charge_abundance'
    ]

    missing_features = [feat for feat in features if feat not in test_features_df.columns]
    if missing_features:
        logger.warning(f"The following features are missing in test_protein_features.csv: {missing_features}")

    # Extract features for FN and TP
    fn_features_df = test_features_df[test_features_df[id_column].isin(fn_ids)]
    tp_features_df = test_features_df[test_features_df[id_column].isin(tp_ids)]

    for feature in features:
        if feature not in test_features_df.columns:
            logger.warning(f"Feature '{feature}' is missing. Skipping boxplot for this feature.")
            continue  # Skip missing features

        fn_feature_values = fn_features_df[feature].dropna().values
        tp_feature_values = tp_features_df[feature].dropna().values

        if len(fn_feature_values) == 0 or len(tp_feature_values) == 0:
            logger.warning(f"No data available for feature '{feature}' in one of the groups. Skipping boxplot.")
            continue  # Skip if no data in one of the groups

        data = pd.DataFrame({
            feature: np.concatenate([fn_feature_values, tp_feature_values]),
            'Group': ['False Negatives'] * len(fn_feature_values) + ['True Positives'] * len(tp_feature_values)
        })

        # Ensure 'Group' is categorical
        data['Group'] = data['Group'].astype('category')

        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Group', y=feature, data=data)
        plt.title(f'{feature} Distribution: FN vs TP')
        plt.xlabel('Group')
        plt.ylabel(feature)
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{feature}_boxplot_fn_vs_tp.png')
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Saved boxplot for {feature} to {output_file}")

    logger.info("SVM error analysis completed.")
