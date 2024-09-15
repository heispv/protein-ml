from collections import namedtuple

PositiveProteinData = namedtuple('PositiveProteinData', ['scientific_name', 'primary_accession', 'lineage', 'fasta', 'sequence_length', 'ps_length'])
NegativeProteinData = namedtuple('NegativeProteinData', ['scientific_name', 'primary_accession', 'lineage', 'fasta', 'sequence_length', 'tm_helix'])
