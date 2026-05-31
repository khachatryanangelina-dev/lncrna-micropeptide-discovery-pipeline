# lncRNA Micropeptide Discovery Pipeline

A reproducible bioinformatics workflow for the identification, annotation and prioritization of putative micropeptide-coding open reading frames (smORFs) within long non-coding RNA (lncRNA) transcripts expressed in human skeletal muscle.

---

## Overview

Long non-coding RNAs (lncRNAs) are traditionally classified as non-protein-coding transcripts. Recent studies, however, have demonstrated that some lncRNAs contain short open reading frames (smORFs) capable of producing biologically active micropeptides involved in diverse cellular processes, including muscle development, regeneration and metabolic regulation.

Because micropeptides are typically short, weakly conserved and frequently absent from reference protein annotations, their systematic discovery remains challenging.

This project implements a reproducible computational pipeline designed to identify and prioritize candidate micropeptide-coding ORFs from skeletal muscle-expressed lncRNA transcripts using transcriptomic, sequence-derived and contextual features.

---

## Pipeline Highlights

* Tissue-specific filtering of lncRNA transcripts using GTEx skeletal muscle expression data
* Stop-to-stop ORF discovery using EMBOSS getorf
* Comprehensive ORF annotation and feature extraction
* Multi-level transcript-context analysis
* Integrated candidate prioritization framework
* External coding-potential validation using independent prediction tools
* Fully script-based and reproducible workflow

---

## Computational Workflow

```text
GTEx expression data + GENCODE annotation
                    │
                    ▼
      Muscle-specific lncRNA filtering
                    │
                    ▼
      Transcript sequence extraction
                    │
                    ▼
      ORF discovery (EMBOSS getorf)
                    │
                    ▼
           ORF annotation
                    │
                    ▼
         Feature engineering
                    │
                    ▼
  Integrated micropeptide scoring
                    │
                    ▼
      Candidate prioritization
                    │
                    ▼
    External coding-potential assessment
```

---

## Data Sources

The pipeline relies on publicly available reference resources:

### GENCODE v49

Reference human genome annotation and transcript sequences.

### GTEx v11

Transcriptomic expression profiles from human skeletal muscle samples.

Input datasets are not distributed with this repository and must be downloaded independently from their respective providers.

---

## Methodology

### 1. Tissue-Specific Transcript Selection

The analysis begins with identification of lncRNA genes and transcripts expressed in skeletal muscle.

A transcript is retained when expression satisfies:

```text
TPM > 1 in at least 10% of skeletal muscle samples
```

This step restricts the search space to biologically relevant transcripts while preserving low-abundance candidates that may still encode functional micropeptides.

---

### 2. ORF Discovery

Candidate ORFs are detected using EMBOSS `getorf` in stop-to-stop mode.

Unlike conventional protein-coding gene prediction approaches, this strategy does not require a canonical AUG start codon, allowing detection of potentially non-canonical translated smORFs.

---

### 3. ORF Annotation

Each ORF is annotated using a broad collection of sequence-derived and contextual features, including:

* ORF length
* GC content and GC3 content
* nucleotide complexity
* codon usage statistics
* transcript position
* overlap architecture
* transcript-level context
* translated peptide properties

---

### 4. Feature Engineering and Candidate Ranking

Candidate prioritization is performed using an integrated scoring framework that combines:

* sequence integrity metrics
* nucleotide architecture
* ORF structural properties
* transcript context
* overlap independence
* expression support
* peptide-derived characteristics

The resulting:

```text
integrated_micropeptide_score
```

is used to rank ORF candidates and generate representative candidate sets for downstream analysis.

---

## Results

Application of the pipeline to human skeletal muscle transcriptomes produced:

| Metric                                     |              Value |
| ------------------------------------------ | -----------------: |
| Muscle-expressed lncRNA genes              |              2,599 |
| Retained lncRNA transcripts                |              2,322 |
| Detected ORF candidates                    |             54,310 |
| Annotated ORF candidates                   |             54,310 |
| Representative transcript-level candidates |              2,322 |
| Representative gene-level candidates       |              1,314 |
| Final prioritized candidate set            | Top 100 candidates |

The resulting candidate collection represents a computationally prioritized resource for future experimental validation and biological characterization.

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

results/
├── 01_prepare/
├── 02_getorf/
└── ...

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

Install the package:

```bash
pip install -e .
```

---

## Reproducibility

The repository contains the complete source code required to reproduce the computational workflow.

Users must independently obtain:

* GENCODE v49 annotation files
* GENCODE v49 transcript FASTA sequences
* GTEx v11 transcript expression matrices

Input paths and analysis parameters are configured through:

```text
configs/default.yaml
```

---

## Limitations

The presented workflow provides computational prioritization rather than direct evidence of translation.

Candidate ORFs identified by the pipeline should be considered hypotheses requiring independent validation using approaches such as:

* ribosome profiling (Ribo-seq)
* targeted proteomics
* epitope-tagging experiments
* reporter assays
* CRISPR-based perturbation studies

---

## Author

**Angelina Khachatryan**

Diploma Thesis Project

2026

---
