import os
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import numpy as np

# Define the scales as dictionaries
HYDROPATHY_SCALE = { 
    'A': 1.800, 'C': 2.500, 'D': -3.500, 'E': -3.500, 'F': 2.800,
    'G': -0.400, 'H': -3.200, 'I': 4.500, 'K': -3.900, 'L': 3.800,
    'M': 1.900, 'N': -3.500, 'P': -1.600, 'Q': -3.500, 'R': -4.500,
    'S': -0.800, 'T': -0.700, 'V': 4.200, 'W': -0.900, 'Y': -1.300
}

CHARGE_SCALE = { 
    'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0,
    'G': 0.0, 'H': 0.0, 'I': 0.0, 'K': 1.0, 'L': 0.0,
    'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 1.0,
    'S': 0.0, 'T': 0.0, 'V': 0.0, 'W': 0.0, 'Y': 0.0
}

ALPHA_HELIX_PROPENSITY = {
    'A': 1.420, 'C': 0.700, 'D': 1.010, 'E': 1.510, 'F': 1.130,
    'G': 0.570, 'H': 1.000, 'I': 1.080, 'K': 1.160, 'L': 1.210,
    'M': 1.450, 'N': 0.670, 'P': 0.570, 'Q': 1.110, 'R': 0.980,
    'S': 0.770, 'T': 0.830, 'V': 1.060, 'W': 1.080, 'Y': 0.690
}

TRANSMEMBRANE_TENDENCY = {
    'A': 0.380, 'C': -0.300, 'D': -3.270, 'E': -2.900, 'F': 1.980,
    'G': -0.190, 'H': -1.440, 'I': 1.970, 'K': -3.460, 'L': 1.820,
    'M': 1.400, 'N': -1.620, 'P': -1.440, 'Q': -1.840, 'R': -2.570,
    'S': -0.530, 'T': -0.320, 'V': 1.460, 'W': 1.530, 'Y': 0.490
}

# Feature Extraction Functions

def extract_composition(sequence, window=20):
    """
    Extracts the frequency of each amino acid in the first `window` residues.
    Returns a dictionary with 'comp_' prefixed amino acids as keys and their frequencies as values.
    """
    aa_counts = {aa: 0.0 for aa in HYDROPATHY_SCALE.keys()}
    seq_window = sequence[:window]
    total = len(seq_window)
    for aa in seq_window:
        if aa in aa_counts:
            aa_counts[aa] += 1
    # Convert counts to frequencies and add 'comp_' prefix
    composition = {f'comp_{aa}': (aa_counts[aa] / total if total > 0 else 0) for aa in aa_counts}
    return composition

def sliding_window(sequence, window_size):
    """
    Generates a sliding window over the sequence.
    """
    if len(sequence) < window_size:
        return []
    return [sequence[i:i+window_size] for i in range(len(sequence) - window_size + 1)]

def extract_hydrophobicity(sequence, window=40, window_size=5):
    """
    Calculates the maximal and average hydrophobicity using a sliding window of size 5 on the first 40 residues.
    """
    seq_window = sequence[:window]
    hydrophobic_values = [HYDROPATHY_SCALE.get(aa, 0.0) for aa in seq_window]
    windows = sliding_window(hydrophobic_values, window_size)
    if not windows:
        return {'max_hydrophobicity': 0.0, 'avg_hydrophobicity': 0.0}
    window_sums = [sum(w) for w in windows]
    max_hydro = max(window_sums)
    avg_hydro = np.mean(window_sums)
    return {'max_hydrophobicity': max_hydro, 'avg_hydrophobicity': avg_hydro}

def extract_charge_features(sequence, window=3):
    """
    Calculates the maximal abundance of positively-charged residues (K, R) and the normalized position of the max value.
    """
    seq_window = sequence[:40]
    charge_values = [CHARGE_SCALE.get(aa, 0.0) for aa in seq_window]
    windows = sliding_window(charge_values, window)
    if not windows:
        return {'max_charge_abundance': 0.0, 'pos_max_charge_abundance': 0.0}
    window_sums = [sum(w) for w in windows]
    max_charge = max(window_sums)
    if max_charge > 0:
        max_index = window_sums.index(max_charge)
        normalized_pos = (max_index + window//2) / len(windows)
    else:
        normalized_pos = 0.0
    return {'max_charge_abundance': max_charge, 'pos_max_charge_abundance': normalized_pos}

def extract_alpha_helix_propensity(sequence, window=7):
    """
    Calculates the average and maximal alpha-helix propensity using a sliding window of size 7.
    """
    seq_window = sequence[:40]
    propensity_values = [ALPHA_HELIX_PROPENSITY.get(aa, 0.0) for aa in seq_window]
    windows = sliding_window(propensity_values, window)
    if not windows:
        return {'avg_alpha_propensity': 0.0, 'max_alpha_propensity': 0.0}
    window_averages = [np.mean(w) for w in windows]
    avg_propensity = np.mean(window_averages)
    max_propensity = max(window_averages)
    return {'avg_alpha_propensity': avg_propensity, 'max_alpha_propensity': max_propensity}

def extract_transmembrane_propensity(sequence, window=7):
    """
    Calculates the average and maximal transmembrane propensity using a sliding window of size 7.
    """
    seq_window = sequence[:40]
    propensity_values = [TRANSMEMBRANE_TENDENCY.get(aa, 0.0) for aa in seq_window]
    windows = sliding_window(propensity_values, window)
    if not windows:
        return {'avg_transmembrane_propensity': 0.0, 'max_transmembrane_propensity': 0.0}
    window_averages = [np.mean(w) for w in windows]
    avg_propensity = np.mean(window_averages)
    max_propensity = max(window_averages)
    return {'avg_transmembrane_propensity': avg_propensity, 'max_transmembrane_propensity': max_propensity}

def extract_features(sequence):
    """
    Extracts all features for a given sequence.
    """
    # Ensure the sequence is uppercase
    sequence = sequence.upper()
    
    # Validate sequence contains only valid amino acids
    valid_aas = set(HYDROPATHY_SCALE.keys())
    sequence_set = set(sequence)
    invalid_aas = sequence_set - valid_aas
    if invalid_aas:
        print(f"Warning: Sequence contains invalid amino acids: {invalid_aas}. These will be treated as having a scale value of 0.0.")
    
    features = {}
    
    # 1. Composition Frequency
    composition = extract_composition(sequence, window=20)
    features.update(composition)
    
    # 2. Hydrophobicity
    hydrophobicity = extract_hydrophobicity(sequence, window=40, window_size=5)
    features.update(hydrophobicity)
    
    # 3. Charge Features
    charge = extract_charge_features(sequence, window=3)
    features.update(charge)
    
    # 4. Alpha-Helix Propensity
    alpha_helix = extract_alpha_helix_propensity(sequence, window=7)
    features.update(alpha_helix)
    
    # 5. Transmembrane Propensity
    transmembrane = extract_transmembrane_propensity(sequence, window=7)
    features.update(transmembrane)
    
    return features

def parse_fasta_files(pos_dir, neg_dir):
    """
    Parses FASTA files from positive and negative directories and extracts features.
    Returns a pandas DataFrame with all features and labels.
    """
    data = []
    
    # Process positive samples
    pos_files = list(Path(pos_dir).glob("*.fasta"))
    for filepath in pos_files:
        for record in SeqIO.parse(filepath, "fasta"):
            seq = str(record.seq).upper()
            accession_id = record.id
            features = extract_features(seq)
            features['accession_id'] = accession_id
            features['label'] = 1
            data.append(features)
    
    # Process negative samples
    neg_files = list(Path(neg_dir).glob("*.fasta"))
    for filepath in neg_files:
        for record in SeqIO.parse(filepath, "fasta"):
            seq = str(record.seq).upper()
            accession_id = record.id
            features = extract_features(seq)
            features['accession_id'] = accession_id
            features['label'] = 0
            data.append(features)
    
    df = pd.DataFrame(data)
    
    # Reorder columns to have accession_id first and label last
    cols = ['accession_id'] + [col for col in df.columns if col not in ['accession_id', 'label']] + ['label']
    df = df[cols]
    return df

def main():
    # Define directories
    base_dir = Path('data/splited_data/train')  # Ensure this path is correct
    pos_dir = base_dir / 'pos'
    neg_dir = base_dir / 'neg'
    
    # Check if input directories exist
    if not pos_dir.exists():
        raise FileNotFoundError(f"Positive directory not found: {pos_dir}")
    if not neg_dir.exists():
        raise FileNotFoundError(f"Negative directory not found: {neg_dir}")
    
    # Extract features and create DataFrame
    df = parse_fasta_files(pos_dir, neg_dir)
    
    # Display the first few rows
    print(df.head())
    
    # Define the output directory
    output_dir = Path('data/features')
    
    # Create the directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify feature columns (excluding 'accession_id' and 'label')
    feature_cols = [col for col in df.columns if col not in ['accession_id', 'label']]
    
    # Compute mean and std for each feature
    feature_means = df[feature_cols].mean()
    feature_stds = df[feature_cols].std()
    
    # Save mean and std to CSV files
    feature_means.to_csv(output_dir / 'feature_means.csv', header=True)
    feature_stds.to_csv(output_dir / 'feature_stds.csv', header=True)
    
    # Save the DataFrame to a CSV file
    output_file = output_dir / 'protein_features.csv'
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    
    # Normalize the feature columns
    df[feature_cols] = (df[feature_cols] - feature_means) / feature_stds
    
    # Save the DataFrame to a CSV file
    output_file = output_dir / 'norm_protein_features.csv'
    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")
    print(f"Feature means saved to {output_dir / 'feature_means.csv'}")
    print(f"Feature stds saved to {output_dir / 'feature_stds.csv'}")
    
    
    ################## Test Set
    test_base_dir = Path('data/splited_data/test')  # Ensure this path is correct
    test_pos_dir = test_base_dir / 'pos'
    test_neg_dir = test_base_dir / 'neg'
    
    # Check if input directories exist
    if not test_pos_dir.exists():
        raise FileNotFoundError(f"Positive directory not found: {test_pos_dir}")
    if not test_neg_dir.exists():
        raise FileNotFoundError(f"Negative directory not found: {test_neg_dir}")
    
    # Extract features and create DataFrame
    test_df = parse_fasta_files(test_pos_dir, test_neg_dir)
    
    # Display the first few rows
    print(test_df.head())
    
    # Define the output directory
    test_output_dir = Path('data/features/testing/')
    
    # Create the directory if it doesn't exist
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify feature columns (excluding 'accession_id' and 'label')
    test_feature_cols = [col for col in test_df.columns if col not in ['accession_id', 'label']]
    
    # Save the DataFrame to a CSV file
    test_output_file = test_output_dir / 'test_protein_features.csv'
    test_df.to_csv(test_output_file, index=False)
    print(f"Features saved to {test_output_file}")
    
    test_df[test_feature_cols] = (test_df[test_feature_cols] - feature_means) / feature_stds
    
    # Save the DataFrame to a CSV file
    test_output_file = test_output_dir / 'test_norm_protein_features.csv'
    test_df.to_csv(test_output_file, index=False)
    print(f"Features saved to {test_output_file}")

if __name__ == "__main__":
    main()
