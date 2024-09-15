from typing import Dict, Any
from models import PositiveProteinData, NegativeProteinData
import logging
import json
import os
from config import FETCHED_DIR
from pipeline_01_data_fetcher import get_batch
import requests

def extract_features_pos(sample: Dict[str, Any]) -> PositiveProteinData:
    scientific_name = sample['organism']['scientificName']
    primary_accession = sample['primaryAccession']
    sequence = sample['sequence']['value']
    sequence_length = sample['sequence']['length']

    lineage_list = sample['organism']['lineage']
    lineage = next((l for l in ['Metazoa', 'Fungi', 'Viridiplantae'] if l in lineage_list), 'Other')

    ps_length = next((feature['location']['end']['value'] 
                      for feature in sample["features"] 
                      if feature['type'] == 'Signal' and
                      feature['description'] != 'Not cleaved' and
                      feature['location']['end']['value'] > 14),
                     None)

    fasta = f">{primary_accession}\n{sequence}\n"

    return PositiveProteinData(scientific_name, primary_accession, lineage, fasta, sequence_length, ps_length)

def extract_features_neg(sample: Dict[str, Any]) -> NegativeProteinData:
    scientific_name = sample['organism']['scientificName']
    primary_accession = sample['primaryAccession']
    sequence = sample['sequence']['value']
    sequence_length = sample['sequence']['length']

    lineage_list = sample['organism']['lineage']
    lineage = next((l for l in ['Metazoa', 'Fungi', 'Viridiplantae'] if l in lineage_list), 'Other')

    tm_helix = any(feature['type'] == 'Transmembrane' and 
                   'Helical' in feature['description'] and 
                   feature['location']['start']['value'] <= 90 
                   for feature in sample['features'])

    fasta = f">{primary_accession}\n{sequence}\n"

    return NegativeProteinData(scientific_name, primary_accession, lineage, fasta, sequence_length, tm_helix)

def get_pos_dataset(search_url: str, output_base_name: str, session: requests.Session) -> None:
    n_total, n_filtered = 0, 0

    tsv_file_name = os.path.join(FETCHED_DIR, f"{output_base_name}.tsv")
    fasta_file_name = os.path.join(FETCHED_DIR, f"{output_base_name}.fasta")

    logging.info(f"Processing positive dataset. Output files: {tsv_file_name}, {fasta_file_name}")

    with open(tsv_file_name, "w") as tsv_file, open(fasta_file_name, "w") as fasta_file:
        print("scientific_name", "primary_accession", "lineage", "sequence_length", "ps_length", sep="\t", file=tsv_file)

        for batch, total in get_batch(search_url, session):
            batch_json = json.loads(batch.text)
            for entry in batch_json['results']:
                n_total += 1
                protein_data = extract_features_pos(entry)
                if isinstance(protein_data.ps_length, int) and protein_data.ps_length >= 14:
                    n_filtered += 1
                    print(protein_data.scientific_name, protein_data.primary_accession, protein_data.lineage, 
                          protein_data.sequence_length, protein_data.ps_length, sep="\t", file=tsv_file)
                    fasta_file.write(protein_data.fasta)

    logging.info(f"Positive dataset processing complete. Total entries: {n_total}, Filtered entries: {n_filtered}")
    print(f"Total entries processed: {n_total}")
    print(f"Filtered entries: {n_filtered}")
    print(f"TSV file saved as: {tsv_file_name}")
    print(f"FASTA file saved as: {fasta_file_name}")

def get_neg_dataset(search_url: str, output_base_name: str, session: requests.Session) -> None:
    n_total = 0
    n_tm_helix_true = 0
    n_tm_helix_false = 0

    tsv_file_name = os.path.join(FETCHED_DIR, f"{output_base_name}.tsv")
    fasta_file_name = os.path.join(FETCHED_DIR, f"{output_base_name}.fasta")

    logging.info(f"Processing negative dataset. Output files: {tsv_file_name}, {fasta_file_name}")

    with open(tsv_file_name, "w") as tsv_file, open(fasta_file_name, "w") as fasta_file:
        print("scientific_name", "primary_accession", "lineage", "sequence_length", "tm_helix", sep="\t", file=tsv_file)

        for batch, total in get_batch(search_url, session):
            batch_json = json.loads(batch.text)
            for entry in batch_json['results']:
                n_total += 1
                protein_data = extract_features_neg(entry)
                
                if protein_data.tm_helix:
                    n_tm_helix_true += 1
                else:
                    n_tm_helix_false += 1
                
                print(protein_data.scientific_name, protein_data.primary_accession, 
                      protein_data.lineage, protein_data.sequence_length, protein_data.tm_helix, sep="\t", file=tsv_file)
                fasta_file.write(protein_data.fasta)

    logging.info(f"Negative dataset processing complete. Total entries: {n_total}")
    logging.info(f"Proteins with tm_helix true: {n_tm_helix_true}")
    logging.info(f"Proteins with tm_helix false: {n_tm_helix_false}")
    print(f"Total entries processed: {n_total}")
    print(f"Proteins with tm_helix true: {n_tm_helix_true}")
    print(f"Proteins with tm_helix false: {n_tm_helix_false}")
    print(f"TSV file saved as: {tsv_file_name}")
    print(f"FASTA file saved as: {fasta_file_name}")
    