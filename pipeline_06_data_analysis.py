import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter

def analyze_protein_length_distribution(splitted_data_dir, output_dir, max_length=None):
    """
    Analyzes the distribution of protein lengths for positive and negative sequences in training and test sets.
    Generates combined histogram and density plots, and saves them in the specified output directory.

    Args:
        splitted_data_dir (str): Path to the directory containing split data ('train' and 'test' directories).
        output_dir (str): Path to the directory where the plots will be saved.
        max_length (int, optional): Maximum protein length to include in the analysis. Sequences longer than
                                    this length will be filtered out.
    """
    logging.info("Starting protein length distribution analysis.")
    sets = ['train', 'test']
    data_types = ['pos', 'neg']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for set_type in sets:
        length_data = []

        for data_type in data_types:
            tsv_dir = os.path.join(splitted_data_dir, set_type, data_type)
            # Find the .tsv file in the directory
            tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]
            if not tsv_files:
                logging.warning(f"No .tsv files found in {tsv_dir}")
                continue
            tsv_file_path = os.path.join(tsv_dir, tsv_files[0])

            # Read the tsv file
            df = pd.read_csv(tsv_file_path, sep='\t')
            if 'sequence_length' not in df.columns:
                logging.warning(f"'sequence_length' column not found in {tsv_file_path}")
                continue
            # Extract sequence lengths
            lengths = df['sequence_length'].dropna()

            # Filter lengths based on max_length
            if max_length is not None:
                initial_count = len(lengths)
                lengths = lengths[lengths <= max_length]
                filtered_count = len(lengths)
                logging.info(f"Filtered {initial_count - filtered_count} sequences longer than {max_length} for {data_type} in {set_type} set.")

            length_data.append(pd.DataFrame({
                'sequence_length': lengths,
                'label': data_type
            }))

        if not length_data:
            logging.warning(f"No data found for {set_type} set.")
            continue

        # Combine positive and negative data
        combined_data = pd.concat(length_data, ignore_index=True)

        # Check if combined_data is empty
        if combined_data.empty:
            logging.warning(f"No data to plot for {set_type} set after filtering.")
            continue

        # Plotting
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Combined Histogram and Density Plot
        sns.histplot(data=combined_data, x='sequence_length', hue='label', bins=50, stat="density", common_norm=False)
        sns.kdeplot(data=combined_data, x='sequence_length', alpha=0.5, hue='label', common_norm=False)
        plt.title(f'Protein Length Distribution ({set_type.capitalize()} Set)')
        plt.xlabel('Sequence Length')
        plt.ylabel('Density')
        plt.tight_layout()
        combined_output_path = os.path.join(output_dir, f'protein_length_distribution_{set_type}.png')
        plt.savefig(combined_output_path)
        plt.close()
        logging.info(f"Saved combined histogram and density plot to {combined_output_path}")

    logging.info("Protein length distribution analysis completed.")

def analyze_signal_peptide_length_distribution(splitted_data_dir, output_dir, max_length=None):
    """
    Analyzes the distribution of signal peptide lengths (ps_length) for positive sequences in training and test sets.
    Generates combined histogram and density plots, and saves them in the specified output directory.

    Args:
        splitted_data_dir (str): Path to the directory containing split data ('train' and 'test' directories).
        output_dir (str): Path to the directory where the plots will be saved.
        max_length (int, optional): Maximum SP length to include in the analysis. Values longer than
                                    this length will be filtered out.
    """
    logging.info("Starting signal peptide length distribution analysis.")
    sets = ['train', 'test']
    data_type = 'pos'  # Only positive data has 'ps_length'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for set_type in sets:
        tsv_dir = os.path.join(splitted_data_dir, set_type, data_type)
        # Find the .tsv file in the directory
        tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]
        if not tsv_files:
            logging.warning(f"No .tsv files found in {tsv_dir}")
            continue
        tsv_file_path = os.path.join(tsv_dir, tsv_files[0])

        # Read the tsv file
        df = pd.read_csv(tsv_file_path, sep='\t')
        if 'ps_length' not in df.columns:
            logging.warning(f"'ps_length' column not found in {tsv_file_path}")
            continue
        # Extract ps_lengths
        ps_lengths = df['ps_length'].dropna()

        # Convert ps_lengths to numeric, in case they are not
        ps_lengths = pd.to_numeric(ps_lengths, errors='coerce').dropna()

        # Filter lengths based on max_length
        if max_length is not None:
            initial_count = len(ps_lengths)
            ps_lengths = ps_lengths[ps_lengths <= max_length]
            filtered_count = len(ps_lengths)
            logging.info(f"Filtered {initial_count - filtered_count} SP lengths longer than {max_length} in {set_type} set.")

        if ps_lengths.empty:
            logging.warning(f"No data to plot for {set_type} set after filtering.")
            continue

        # Create a DataFrame
        data = pd.DataFrame({'ps_length': ps_lengths})

        # Plotting
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Combined Histogram and Density Plot
        sns.histplot(data=data, x='ps_length', bins=30, stat="density", common_norm=False)
        sns.kdeplot(data=data, x='ps_length', alpha=0.5, common_norm=False)
        plt.title(f'Signal Peptide Length Distribution ({set_type.capitalize()} Set)')
        plt.xlabel('Signal Peptide Length')
        plt.ylabel('Density')
        plt.tight_layout()
        combined_output_path = os.path.join(output_dir, f'sp_length_distribution_{set_type}.png')
        plt.savefig(combined_output_path)
        plt.close()
        logging.info(f"Saved combined histogram and density plot to {combined_output_path}")

    logging.info("Signal peptide length distribution analysis completed.")

def compare_amino_acid_composition(splitted_data_dir, output_dir):
    """
    Compares the amino acid composition of SP sequences against the SwissProt background.
    Generates bar plots comparing the AA compositions and saves them in the specified output directory.

    Args:
        splitted_data_dir (str): Path to the directory containing split data ('train' and 'test' directories).
        output_dir (str): Path to the directory where the plots will be saved.
    """
    logging.info("Starting comparative amino acid composition analysis.")
    sets = ['train', 'test']
    data_type = 'pos'  # Only positive data has SP sequences

    # SwissProt AA composition
    swissprot_aa_composition = {
        'A': 8.25, 'R': 5.52, 'N': 4.06, 'D': 5.46, 'C': 1.38,
        'Q': 3.93, 'E': 6.71, 'G': 7.07, 'H': 2.27, 'I': 5.91,
        'L': 9.64, 'K': 5.80, 'M': 2.41, 'F': 3.86, 'P': 4.74,
        'S': 6.65, 'T': 5.36, 'W': 1.10, 'Y': 2.92, 'V': 6.85
    }
    standard_amino_acids = set(swissprot_aa_composition.keys())

    # Desired order of amino acids
    amino_acids_order = ['G', 'A', 'V', 'P', 'L', 'I', 'M', 'F', 'W', 'Y', 'S', 'T', 'C', 'N', 'Q', 'H', 'D', 'E', 'K', 'R']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for set_type in sets:
        tsv_dir = os.path.join(splitted_data_dir, set_type, data_type)
        # Find the .tsv file in the directory
        tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]
        if not tsv_files:
            logging.warning(f"No .tsv files found in {tsv_dir}")
            continue
        tsv_file_path = os.path.join(tsv_dir, tsv_files[0])

        # Find the .fasta file in the directory
        fasta_files = [f for f in os.listdir(tsv_dir) if f.endswith('.fasta')]
        if not fasta_files:
            logging.warning(f"No .fasta files found in {tsv_dir}")
            continue
        fasta_file_path = os.path.join(tsv_dir, fasta_files[0])

        # Read the .tsv file
        df = pd.read_csv(tsv_file_path, sep='\t')
        if 'primary_accession' not in df.columns or 'ps_length' not in df.columns:
            logging.warning(f"'primary_accession' or 'ps_length' columns not found in {tsv_file_path}")
            continue

        # Build a dictionary mapping primary_accession to ps_length
        id_to_ps_length = dict(zip(df['primary_accession'], df['ps_length']))

        # Read the .fasta file and build a dictionary mapping primary_accession to sequence
        fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file_path, 'fasta'))

        # Collect SP sequences
        sp_sequences = []
        for primary_accession, ps_length in id_to_ps_length.items():
            if primary_accession in fasta_sequences:
                seq = str(fasta_sequences[primary_accession].seq)
                if pd.isna(ps_length):
                    continue
                # Ensure ps_length is integer
                try:
                    ps_length = int(ps_length)
                except ValueError:
                    continue
                # Extract SP sequence
                sp_seq = seq[:ps_length]
                sp_sequences.append(sp_seq)
            else:
                logging.warning(f"Primary accession {primary_accession} not found in fasta file.")

        if not sp_sequences:
            logging.warning(f"No SP sequences found for {set_type} set.")
            continue

        # Concatenate all SP sequences
        all_sp_seq = ''.join(sp_sequences)

        # Count amino acids, include only standard amino acids
        aa_counts = Counter(aa for aa in all_sp_seq if aa in standard_amino_acids)

        # Calculate percentages
        total_aa = sum(aa_counts.values())
        aa_percentages = {aa: (aa_counts.get(aa, 0) / total_aa * 100) for aa in standard_amino_acids}

        # Ensure all amino acids are represented
        for aa in standard_amino_acids:
            aa_percentages.setdefault(aa, 0.0)

        # Prepare data for plotting
        amino_acids = amino_acids_order  # Use the specified order
        our_data_percentages = [aa_percentages.get(aa, 0.0) for aa in amino_acids]
        swissprot_percentages = [swissprot_aa_composition.get(aa, 0.0) for aa in amino_acids]

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Amino Acid': amino_acids * 2,
            'Percentage': our_data_percentages + swissprot_percentages,
            'Source': ['Our SP Sequences'] * len(amino_acids) + ['SwissProt'] * len(amino_acids)
        })

        # Plotting
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))

        # Define the color palette
        palette = {'Our SP Sequences': 'blue', 'SwissProt': 'red'}

        sns.barplot(
            x='Amino Acid',
            y='Percentage',
            hue='Source',
            data=plot_df,
            order=amino_acids,
            palette=palette
        )
        plt.title(f'Amino Acid Composition Comparison ({set_type.capitalize()} Set)')
        plt.xlabel('Amino Acid')
        plt.ylabel('Percentage')
        plt.legend(title='Source')
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'aa_composition_comparison_{set_type}.png')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved amino acid composition comparison plot to {output_path}")

    logging.info("Comparative amino acid composition analysis completed.")
    
def plot_taxonomic_classification(splitted_data_dir, output_dir, num_classifications=5):
    """
    Plots the taxonomic classification distribution as pie charts for training and test sets,
    combining both positive and negative data.

    Args:
        splitted_data_dir (str): Path to the directory containing split data ('train' and 'test' directories).
        output_dir (str): Path to the directory where the plots will be saved.
        num_classifications (int): Number of top classifications to display in the pie chart.
    """
    logging.info("Starting taxonomic classification analysis.")
    sets = ['train', 'test']
    data_types = ['pos', 'neg']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for set_type in sets:
        lineage_counts = {}
        for data_type in data_types:
            tsv_dir = os.path.join(splitted_data_dir, set_type, data_type)
            # Find the .tsv file in the directory
            tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]
            if not tsv_files:
                logging.warning(f"No .tsv files found in {tsv_dir}")
                continue
            tsv_file_path = os.path.join(tsv_dir, tsv_files[0])

            # Read the tsv file
            df = pd.read_csv(tsv_file_path, sep='\t')
            if 'lineage' not in df.columns:
                logging.warning(f"'lineage' column not found in {tsv_file_path}")
                continue

            # Count occurrences of each lineage
            lineage_series = df['lineage'].dropna()
            counts = lineage_series.value_counts().to_dict()

            # Combine counts
            for lineage, count in counts.items():
                lineage_counts[lineage] = lineage_counts.get(lineage, 0) + count

        # If no data, continue to next set
        if not lineage_counts:
            logging.warning(f"No lineage data found for {set_type} set.")
            continue

        # Sort the lineages by count
        sorted_lineages = sorted(lineage_counts.items(), key=lambda x: x[1], reverse=True)

        # Take the top 'num_classifications' lineages
        top_lineages = dict(sorted_lineages[:num_classifications])

        # Sum counts for 'Other' lineages
        other_count = sum(count for lineage, count in sorted_lineages[num_classifications:])
        if other_count > 0:
            top_lineages['Other'] = other_count

        # Prepare data for plotting
        labels = list(top_lineages.keys())
        sizes = list(top_lineages.values())

        # Plotting
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Taxonomic Classification ({set_type.capitalize()} Set)')
        output_path = os.path.join(output_dir, f'taxonomic_classification_{set_type}.png')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved taxonomic classification pie chart to {output_path}")

    logging.info("Taxonomic classification analysis completed.")
    
def plot_scientific_name_classification(splitted_data_dir, output_dir, num_classifications=5):
    """
    Plots the scientific name classification distribution as pie charts for training and test sets,
    combining both positive and negative data.

    Args:
        splitted_data_dir (str): Path to the directory containing split data ('train' and 'test' directories).
        output_dir (str): Path to the directory where the plots will be saved.
        num_classifications (int): Number of top scientific names to display in the pie chart.
    """
    logging.info("Starting scientific name classification analysis.")
    sets = ['train', 'test']
    data_types = ['pos', 'neg']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for set_type in sets:
        scientific_name_counts = {}
        for data_type in data_types:
            tsv_dir = os.path.join(splitted_data_dir, set_type, data_type)
            # Find the .tsv file in the directory
            tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]
            if not tsv_files:
                logging.warning(f"No .tsv files found in {tsv_dir}")
                continue
            tsv_file_path = os.path.join(tsv_dir, tsv_files[0])

            # Read the .tsv file
            df = pd.read_csv(tsv_file_path, sep='\t')
            if 'scientific_name' not in df.columns:
                logging.warning(f"'scientific_name' column not found in {tsv_file_path}")
                continue

            # Count occurrences of each scientific_name
            scientific_name_series = df['scientific_name'].dropna()
            counts = scientific_name_series.value_counts().to_dict()

            # Combine counts
            for sci_name, count in counts.items():
                scientific_name_counts[sci_name] = scientific_name_counts.get(sci_name, 0) + count

        # If no data, continue to next set
        if not scientific_name_counts:
            logging.warning(f"No scientific_name data found for {set_type} set.")
            continue

        # Sort the scientific_names by count
        sorted_scientific_names = sorted(scientific_name_counts.items(), key=lambda x: x[1], reverse=True)

        # Take the top 'num_classifications' scientific_names
        top_scientific_names = dict(sorted_scientific_names[:num_classifications])

        # Sum counts for 'Other' scientific_names
        other_count = sum(count for sci_name, count in sorted_scientific_names[num_classifications:])
        if other_count > 0:
            top_scientific_names['Other'] = other_count

        # Prepare data for plotting
        labels = list(top_scientific_names.keys())
        sizes = list(top_scientific_names.values())

        # Plotting
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Scientific Name Classification ({set_type.capitalize()} Set)')
        output_path = os.path.join(output_dir, f'scientific_name_classification_{set_type}.png')
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Saved scientific name classification pie chart to {output_path}")

    logging.info("Scientific name classification analysis completed.")
    
def extract_cleavage_site_sequences(splitted_data_dir, output_dir):
    """
    Extracts cleavage site sequences for Multiple WebLogo based on ps_length.
    For each sequence, extracts 12 AA before the cleavage site, the cleavage site AA, and 2 AA after,
    totaling 15 AA. Saves the extracted sequences in FASTA format suitable for WebLogo tools.

    Args:
        splitted_data_dir (str): Path to the directory containing split data ('train' and 'test' directories).
        output_dir (str): Path to the directory where the extracted sequences will be saved.
    """
    logging.info("Starting cleavage site sequence extraction for WebLogo.")
    sets = ['train', 'test']
    data_type = 'pos'  # Only positive data has ps_length

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for set_type in sets:
        tsv_dir = os.path.join(splitted_data_dir, set_type, data_type)
        # Find the .tsv file in the directory
        tsv_files = [f for f in os.listdir(tsv_dir) if f.endswith('.tsv')]
        if not tsv_files:
            logging.warning(f"No .tsv files found in {tsv_dir}")
            continue
        tsv_file_path = os.path.join(tsv_dir, tsv_files[0])

        # Find the .fasta file in the directory
        fasta_files = [f for f in os.listdir(tsv_dir) if f.endswith('.fasta')]
        if not fasta_files:
            logging.warning(f"No .fasta files found in {tsv_dir}")
            continue
        fasta_file_path = os.path.join(tsv_dir, fasta_files[0])

        # Read the .tsv file
        df = pd.read_csv(tsv_file_path, sep='\t')
        if 'primary_accession' not in df.columns or 'ps_length' not in df.columns:
            logging.warning(f"'primary_accession' or 'ps_length' columns not found in {tsv_file_path}")
            continue

        # Build a dictionary mapping primary_accession to ps_length
        id_to_ps_length = dict(zip(df['primary_accession'], df['ps_length']))

        # Read the .fasta file and build a dictionary mapping primary_accession to sequence
        try:
            fasta_sequences = SeqIO.to_dict(SeqIO.parse(fasta_file_path, 'fasta'))
        except Exception as e:
            logging.error(f"Error parsing fasta file {fasta_file_path}: {e}")
            continue

        # Collect extracted sequences
        extracted_sequences = []
        for primary_accession, ps_length in id_to_ps_length.items():
            if primary_accession in fasta_sequences:
                seq_record = fasta_sequences[primary_accession]
                seq = seq_record.seq  # This is a Seq object
                if pd.isna(ps_length):
                    logging.warning(f"ps_length is NaN for {primary_accession} in {tsv_file_path}")
                    continue
                # Ensure ps_length is integer
                try:
                    ps_length = int(ps_length)
                except ValueError:
                    logging.warning(f"Invalid ps_length '{ps_length}' for {primary_accession} in {tsv_file_path}")
                    continue
                # Calculate indices for extraction (adjusted for 1-based index)
                start_index = ps_length - 1 - 12
                end_index = ps_length - 1 + 2  # Exclusive in Python slicing

                # Handle edge cases
                if start_index < 0:
                    logging.warning(f"ps_length {ps_length} too small for {primary_accession}. Skipping.")
                    continue
                if end_index > len(seq):
                    logging.warning(f"ps_length {ps_length} + 2 exceeds sequence length for {primary_accession}. Skipping.")
                    continue

                # Extract the 15 AA window
                cleavage_site_seq = seq[start_index:end_index + 1]  # +1 to include the end_index
                if len(cleavage_site_seq) != 15:
                    logging.warning(f"Extracted sequence length {len(cleavage_site_seq)} != 15 for {primary_accession}. Skipping.")
                    continue

                # Create a SeqRecord
                record = SeqRecord(
                    Seq(str(cleavage_site_seq)),
                    id=f"{primary_accession}_{ps_length}",
                    description=""
                )
                extracted_sequences.append(record)
            else:
                logging.warning(f"Primary accession {primary_accession} not found in fasta file {fasta_file_path}.")

        if not extracted_sequences:
            logging.warning(f"No cleavage site sequences extracted for {set_type} set.")
            continue

        # Define output file path
        output_file = os.path.join(output_dir, f'cleavage_site_sequences_{set_type}.fasta')

        # Write the extracted sequences to a FASTA file
        try:
            SeqIO.write(extracted_sequences, output_file, 'fasta')
            logging.info(f"Saved {len(extracted_sequences)} cleavage site sequences to {output_file}")
        except Exception as e:
            logging.error(f"Error writing to FASTA file {output_file}: {e}")

    logging.info("Cleavage site sequence extraction for WebLogo completed.")