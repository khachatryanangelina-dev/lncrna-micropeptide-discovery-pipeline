# lncRNA Micropeptide Discovery Pipeline

Computational pipeline for identification, annotation and prioritization of putative micropeptide-encoding open reading frames (smORFs) within long non-coding RNA (lncRNA) transcripts expressed in human skeletal muscle.

---

## Project Overview

Long non-coding RNAs (lncRNAs) have traditionally been classified as non-protein-coding transcripts. However, growing evidence indicates that a subset of lncRNAs contains short open reading frames (smORFs) capable of producing biologically active micropeptides.

The discovery of such hidden coding potential remains challenging because micropeptides are typically:

* shorter than conventional proteins;
* poorly conserved across species;
* frequently absent from reference protein annotations;
* difficult to detect experimentally.

This project implements a reproducible computational workflow designed to identify and prioritize candidate micropeptide-coding ORFs within skeletal muscle-expressed lncRNA transcripts.

---

## Research Objectives

The primary objective of the study was to develop a reproducible bioinformatics pipeline for large-scale discovery of lncRNA-derived micropeptide candidates.

The workflow was designed to:

* identify skeletal muscle-expressed lncRNA transcripts;
* detect candidate smORFs;
* annotate sequence and transcript-context features;
* construct a multidimensional candidate scoring framework;
* prioritize the most biologically plausible micropeptide candidates for downstream validation.

---

## Data Sources

The analysis relies on publicly available resources:

### GTEx v11

Human transcriptomic expression profiles from skeletal muscle samples.

### GENCODE v49

Reference genome annotation and transcript sequence collection.

These datasets are not distributed with the repository and must be downloaded independently.

---

## Computational Workflow

### Stage 1 — Tissue-Specific lncRNA Selection

Identification of lncRNA genes and transcripts expressed in human skeletal muscle based on GTEx expression profiles.

### Stage 2 — ORF Discovery

Detection of candidate open reading frames using EMBOSS `getorf`.

### Stage 3 — ORF Annotation

Computation of structural and sequence-derived features, including:

* ORF length;
* GC-content and GC3-content;
* nucleotide complexity metrics;
* codon composition statistics;
* transcript-context characteristics;
* overlap architecture.

### Stage 4 — Candidate Prioritization

Construction of an integrated scoring framework combining:

* sequence integrity metrics;
* nucleotide architecture;
* transcript context;
* overlap independence;
* expression support;
* peptide-derived properties.

Candidates are ranked using the integrated micropeptide score.

---

## Key Results

Application of the pipeline to human skeletal muscle transcriptomes produced:

| Analysis stage                  | Result                             |
| ------------------------------- | ---------------------------------- |
| Muscle-expressed lncRNA genes   | 2,599                              |
| Candidate lncRNA transcripts    | 2,322                              |
| Detected ORF candidates         | 54,310                             |
| Annotated ORF candidates        | 54,310                             |
| Final prioritized candidate set | Top-ranked micropeptide candidates |

The resulting candidate collection represents a resource for future experimental validation and functional characterization of previously unannotated micropeptides.

---

## Repository Structure

```text
configs/
└── default.yaml

src/
└── lncrna_micropeptides/
    ├── cli.py
    ├── pipeline_config.py
    └── stages/
        ├── prepare_data.py
        ├── run_getorf.py
        ├── annotate_orfs.py
        └── score_micropeptides.py

pyproject.toml
README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/khachatryanangelina-dev/lncrna-micropeptide-discovery-pipeline.git
cd lncrna-micropeptide-discovery-pipeline
```

Install dependencies:

```bash
pip install -e .
```

---

## Reproducibility

The repository contains the complete source code required to reproduce the computational workflow.

To execute the pipeline, users must independently obtain:

* GTEx v11 expression matrices;
* GENCODE v49 genome annotation;
* GENCODE v49 transcript FASTA sequences.

Pipeline parameters and file locations are configured through the project configuration files.

---

## Limitations

The presented workflow provides computational prioritization of candidate smORFs and does not constitute direct evidence of translation in vivo.

Experimental validation approaches such as ribosome profiling, targeted proteomics, CRISPR-based perturbation studies, and reporter assays are required to confirm biological functionality.

---


## Author

Angelina Khachatryan

Diploma Thesis Project

2026

---

## License

MIT License

