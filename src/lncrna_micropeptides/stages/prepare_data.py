#!/usr/bin/env python3
from __future__ import annotations
import argparse
import gzip
import logging
import math
import re
from pathlib import Path
from typing import Iterator
import pandas as pd
from lncrna_micropeptides.pipeline_config import get_nested, load_yaml_config, resolve_path

LOG = logging.getLogger("prepare_lncrna_muscle_data")

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

def strip_version(identifier: str | None) -> str | None:
    if identifier is None or pd.isna(identifier):
        return None
    return str(identifier).split(".")[0]

def open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")

def parse_gtf_attributes(attr_string: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for item in attr_string.strip().split(";"):
        item = item.strip()
        if not item:
            continue
        match = re.match(r'(\S+)\s+"([^"]+)"', item)
        if match:
            key, value = match.groups()
            attrs[key] = value
    return attrs

def load_gtex_gene_tpm(path: Path) -> tuple[pd.DataFrame, list[str]]:
    LOG.info("Loading GTEx gene TPM: %s", path)
    df = pd.read_csv(path, sep="\t", skiprows=2)
    required = {"Name", "Description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Gene TPM file is missing required columns: {sorted(missing)}")

    df = df.rename(columns={"Name": "gene_id", "Description": "gene_name"})
    df["gene_id_clean"] = df["gene_id"].map(strip_version)

    sample_cols = [c for c in df.columns if c not in {"gene_id", "gene_name", "gene_id_clean"}]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce")

    LOG.info("Gene TPM table loaded: %s rows x %s columns", f"{len(df):,}", len(df.columns))
    LOG.info("Detected %s GTEx sample columns in gene table", f"{len(sample_cols):,}")
    return df, sample_cols

def load_sample_ids_for_tissue(sample_attr_file: Path, tissue: str) -> list[str]:
    LOG.info("Loading GTEx sample attributes: %s", sample_attr_file)
    sample_attr = pd.read_csv(sample_attr_file, sep="\t")
    required = {"SMTSD", "SAMPID"}
    missing = required - set(sample_attr.columns)
    if missing:
        raise ValueError(f"Sample attribute file is missing required columns: {sorted(missing)}")

    sample_ids = sample_attr.loc[sample_attr["SMTSD"] == tissue, "SAMPID"].astype(str).tolist()
    if not sample_ids:
        raise ValueError(f"No samples found for tissue: {tissue}")

    LOG.info("Found %s samples for tissue '%s'", f"{len(sample_ids):,}", tissue)
    return sample_ids

def resolve_tissue_sample_ids(sample_attr: Path | None, tissue: str, gene_sample_cols: list[str]) -> list[str]:
    if sample_attr is None:
        LOG.info("--sample-attr was not provided; using sample columns from the gene TPM file as the tissue sample set.")
        return list(gene_sample_cols)
    return load_sample_ids_for_tissue(sample_attr, tissue)

def build_gencode_gene_table(gtf_file: Path) -> pd.DataFrame:
    LOG.info("Parsing gene annotations from GTF: %s", gtf_file)
    rows: list[dict[str, object]] = []
    with open_text(gtf_file) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs_raw = fields
            attrs = parse_gtf_attributes(attrs_raw)
            gene_id_versioned = attrs.get("gene_id")
            rows.append(
                {
                    "gene_id": gene_id_versioned,
                    "gene_id_clean": strip_version(gene_id_versioned),
                    "gene_name_gencode": attrs.get("gene_name"),
                    "gene_type": attrs.get("gene_type", attrs.get("gene_biotype")),
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                }
            )

    gene_df = pd.DataFrame(rows).drop_duplicates()
    if gene_df.empty:
        raise ValueError("No gene records were parsed from the GTF file.")
    LOG.info("Parsed %s gene records from GTF", f"{len(gene_df):,}")
    return gene_df


def build_gencode_transcript_table(gtf_file: Path) -> pd.DataFrame:
    LOG.info("Parsing transcript annotations from GTF: %s", gtf_file)
    rows: list[dict[str, object]] = []
    with open_text(gtf_file) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs_raw = fields
            attrs = parse_gtf_attributes(attrs_raw)
            transcript_id_versioned = attrs.get("transcript_id")
            gene_id_versioned = attrs.get("gene_id")
            if not transcript_id_versioned or not gene_id_versioned:
                continue
            rows.append(
                {
                    "transcript_id_versioned": transcript_id_versioned,
                    "transcript_id": strip_version(transcript_id_versioned),
                    "gene_id_versioned": gene_id_versioned,
                    "gene_id": strip_version(gene_id_versioned),
                    "transcript_name": attrs.get("transcript_name"),
                    "gene_name": attrs.get("gene_name"),
                    "transcript_type": attrs.get("transcript_type", attrs.get("transcript_biotype")),
                    "gene_type": attrs.get("gene_type", attrs.get("gene_biotype")),
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                }
            )

    tx_df = pd.DataFrame(rows).drop_duplicates()
    if tx_df.empty:
        raise ValueError("No transcript records were parsed from the GTF file.")
    LOG.info("Parsed %s transcript records from GTF", f"{len(tx_df):,}")
    return tx_df


def add_expression_metrics(df: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["mean_tpm"] = out[sample_cols].mean(axis=1)
    out["median_tpm"] = out[sample_cols].median(axis=1)
    out["max_tpm"] = out[sample_cols].max(axis=1)

    for thr in (0.1, 0.5, 1.0):
        safe_thr = str(thr).replace(".", "_")
        n_col = f"n_samples_gt_{safe_thr}"
        frac_col = f"frac_samples_gt_{safe_thr}"
        out[n_col] = (out[sample_cols] > thr).sum(axis=1)
        out[frac_col] = out[n_col] / len(sample_cols)

    return out


def detect_transcript_tpm_columns(tpm_file: Path) -> tuple[str, str | None, list[str]]:
    with open_text(tpm_file) as fh:
        header = fh.readline().rstrip("\n").split("\t")

    candidates_tx = ["transcript_id", "Name", "transcript", "tx_id"]
    candidates_gene = ["gene_id", "gene", "parent_gene_id"]

    transcript_col = next((c for c in candidates_tx if c in header), None)
    gene_col = next((c for c in candidates_gene if c in header), None)

    if transcript_col is None:
        transcript_col = header[0]
        LOG.warning(
            "Could not identify transcript ID column by name. Falling back to first column: %s",
            transcript_col,
        )

    non_sample_cols = {transcript_col}
    if gene_col:
        non_sample_cols.add(gene_col)
    if "Description" in header:
        non_sample_cols.add("Description")

    sample_cols = [c for c in header if c not in non_sample_cols]
    LOG.info(
        "Transcript TPM columns detected: transcript_col=%s, gene_col=%s, sample_cols=%s",
        transcript_col,
        gene_col,
        f"{len(sample_cols):,}",
    )
    return transcript_col, gene_col, sample_cols


def load_transcript_tpm_for_selected_transcripts(
    tpm_file: Path,
    transcript_ids_to_keep: set[str],
    sample_ids_to_keep: list[str],
    chunksize: int = 5000,
) -> pd.DataFrame:
    transcript_col, gene_col, available_tpm_sample_cols = detect_transcript_tpm_columns(tpm_file)
    selected_samples = [c for c in sample_ids_to_keep if c in available_tpm_sample_cols]
    if not selected_samples:
        raise ValueError("None of the requested tissue sample IDs are present in the transcript TPM file.")

    usecols = [transcript_col] + ([gene_col] if gene_col else []) + selected_samples
    LOG.info("Reading transcript TPM in chunks; keeping %s samples", f"{len(selected_samples):,}")

    reader = pd.read_csv(
        tpm_file,
        sep="\t",
        compression="gzip" if str(tpm_file).endswith(".gz") else None,
        usecols=usecols,
        chunksize=chunksize,
    )

    kept_chunks: list[pd.DataFrame] = []
    total_rows = 0
    total_kept = 0

    for i, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)
        chunk = chunk.rename(columns={transcript_col: "transcript_id"})
        chunk["transcript_id"] = chunk["transcript_id"].map(strip_version)
        if gene_col:
            chunk = chunk.rename(columns={gene_col: "gene_id_from_tpm"})
            chunk["gene_id_from_tpm"] = chunk["gene_id_from_tpm"].map(strip_version)
        chunk = chunk.loc[chunk["transcript_id"].isin(transcript_ids_to_keep)].copy()
        if not chunk.empty:
            kept_chunks.append(chunk)
            total_kept += len(chunk)
        if i == 1 or i % 10 == 0:
            LOG.info("Transcript TPM chunk %s processed: scanned=%s kept=%s", i, f"{total_rows:,}", f"{total_kept:,}")

    if not kept_chunks:
        raise ValueError("No selected transcripts were found in the transcript TPM matrix.")

    out = pd.concat(kept_chunks, ignore_index=True)
    LOG.info("Transcript TPM subset ready: %s rows", f"{len(out):,}")
    return out


def compute_prevalence(expr_df: pd.DataFrame, sample_cols: list[str], threshold: float) -> pd.Series:
    return (expr_df[sample_cols] > threshold).sum(axis=1) / len(sample_cols)


def summarize_transcripts(expr_df: pd.DataFrame, sample_cols: list[str]) -> pd.DataFrame:
    out = expr_df.copy()
    out["median_tpm"] = out[sample_cols].median(axis=1)
    out["mean_tpm"] = out[sample_cols].mean(axis=1)
    out["max_tpm"] = out[sample_cols].max(axis=1)
    out["prevalence_tpm_gt_0_1"] = compute_prevalence(out, sample_cols, 0.1)
    out["prevalence_tpm_gt_0_5"] = compute_prevalence(out, sample_cols, 0.5)
    out["prevalence_tpm_gt_1"] = compute_prevalence(out, sample_cols, 1.0)
    out["n_samples_gt_1"] = (out[sample_cols] > 1.0).sum(axis=1)
    return out


def iterate_fasta(path: Path) -> Iterator[tuple[str, str]]:
    with open_text(path) as fh:
        header = None
        seq_lines: list[str] = []
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line)
        if header is not None:
            yield header, "".join(seq_lines)

def write_wrapped_fasta_record(handle, header: str, sequence: str, width: int = 80) -> None:
    handle.write(f">{header}\n")
    for i in range(0, len(sequence), width):
        handle.write(sequence[i : i + width] + "\n")

def extract_selected_fasta(
    fasta_path: Path,
    selected_versioned_ids: set[str],
    output_fasta: Path,
) -> tuple[int, int, set[str]]:
    LOG.info("Extracting selected transcript FASTA records: %s", fasta_path)
    found_ids: set[str] = set()
    n_total = 0
    n_written = 0
    with open(output_fasta, "w") as out_fh:
        for header, seq in iterate_fasta(fasta_path):
            n_total += 1
            fasta_id = header.split("|")[0].split()[0]
            if fasta_id in selected_versioned_ids:
                write_wrapped_fasta_record(out_fh, header, seq)
                found_ids.add(fasta_id)
                n_written += 1
    LOG.info("FASTA extraction complete: scanned=%s written=%s missing=%s", f"{n_total:,}", f"{n_written:,}", f"{len(selected_versioned_ids - found_ids):,}")
    return n_total, n_written, found_ids

def prepare_data(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gtex_gene, gene_sample_cols = load_gtex_gene_tpm(args.gene_tpm)
    gencode_genes = build_gencode_gene_table(args.gtf)
    lnc_genes = gencode_genes.loc[gencode_genes["gene_type"].isin(set(args.lncrna_biotypes))].copy()
    if lnc_genes.empty:
        raise ValueError("No requested lncRNA genes found in GTF gene annotation.")
    LOG.info("Selected genes in GENCODE gene annotation after biotype filter: %s", f"{len(lnc_genes):,}")

    gtex_lnc_genes = gtex_gene.merge(
        lnc_genes,
        on="gene_id_clean",
        how="inner",
        suffixes=("", "_gencode"),
    )
    LOG.info("GTEx gene table intersected with selected GENCODE genes: %s genes", f"{len(gtex_lnc_genes):,}")

    gene_metrics = add_expression_metrics(gtex_lnc_genes, gene_sample_cols)
    min_samples_gene = max(1, int(math.ceil(len(gene_sample_cols) * args.gene_min_fraction)))
    if args.gene_tpm_threshold == 0.1:
        gene_count_col = "n_samples_gt_0_1"
        gene_frac_col = "frac_samples_gt_0_1"
    elif args.gene_tpm_threshold == 0.5:
        gene_count_col = "n_samples_gt_0_5"
        gene_frac_col = "frac_samples_gt_0_5"
    elif args.gene_tpm_threshold == 1.0:
        gene_count_col = "n_samples_gt_1_0"
        gene_frac_col = "frac_samples_gt_1_0"
    else:
        raise ValueError("gene-tpm-threshold must be one of: 0.1, 0.5, 1.0")

    gene_mask = gene_metrics[gene_count_col] >= min_samples_gene
    filtered_genes = gene_metrics.loc[gene_mask].copy()
    LOG.info(
        "Gene filter applied: TPM > %.3f in >= %.1f%% of samples -> min_samples=%s, retained_genes=%s",
        args.gene_tpm_threshold,
        args.gene_min_fraction * 100,
        min_samples_gene,
        f"{len(filtered_genes):,}",
    )

    gene_cols = [
        "gene_id",
        "gene_id_clean",
        "gene_name",
        "gene_name_gencode",
        "gene_type",
        "chrom",
        "start",
        "end",
        "strand",
        "mean_tpm",
        "median_tpm",
        "max_tpm",
        "n_samples_gt_0_1",
        "frac_samples_gt_0_1",
        "n_samples_gt_0_5",
        "frac_samples_gt_0_5",
        "n_samples_gt_1_0",
        "frac_samples_gt_1_0",
    ]
    filtered_genes = filtered_genes[gene_cols].sort_values(
        [gene_frac_col, "median_tpm", "mean_tpm"],
        ascending=False,
    )

    filtered_genes_file = args.output_dir / "filtered_lncRNA_genes_muscle.tsv"
    filtered_genes.to_csv(filtered_genes_file, sep="\t", index=False)
    LOG.info("Saved filtered genes: %s", filtered_genes_file)

    tissue_sample_ids = resolve_tissue_sample_ids(args.sample_attr, args.tissue, gene_sample_cols)
    tx_table = build_gencode_transcript_table(args.gtf)
    candidate_tx = tx_table.loc[tx_table["gene_id"].isin(set(filtered_genes["gene_id_clean"]))].copy()
    candidate_tx = candidate_tx.loc[candidate_tx["transcript_type"].isin(set(args.lncrna_biotypes))].copy()
    if candidate_tx.empty:
        raise ValueError("No candidate transcripts found for the filtered gene set and requested biotypes.")
    LOG.info("Candidate transcripts from retained genes: %s", f"{len(candidate_tx):,}")

    transcript_tpm = load_transcript_tpm_for_selected_transcripts(
        tpm_file=args.transcript_tpm,
        transcript_ids_to_keep=set(candidate_tx["transcript_id"]),
        sample_ids_to_keep=tissue_sample_ids,
        chunksize=args.chunksize,
    )

    sample_cols = [c for c in tissue_sample_ids if c in transcript_tpm.columns]
    transcript_expr = candidate_tx.merge(transcript_tpm, on="transcript_id", how="inner")
    if "gene_id_from_tpm" in transcript_expr.columns:
        mismatch = (transcript_expr["gene_id"] != transcript_expr["gene_id_from_tpm"]).sum()
        if mismatch:
            LOG.warning("Gene ID mismatches between GTF and transcript TPM subset: %s", f"{mismatch:,}")

    transcript_summary = summarize_transcripts(transcript_expr, sample_cols)
    transcript_summary = transcript_summary.sort_values(
        ["prevalence_tpm_gt_1", "median_tpm", "mean_tpm"],
        ascending=False,
    )

    transcript_summary_file = args.output_dir / "muscle_selected_lncRNA_transcript_summary.tsv"
    transcript_summary.to_csv(transcript_summary_file, sep="\t", index=False)
    LOG.info("Saved transcript summary: %s", transcript_summary_file)

    final_mask = (
        (transcript_summary["median_tpm"] > args.transcript_median_tpm_min)
        & (transcript_summary["prevalence_tpm_gt_1"] >= args.transcript_prevalence_min)
    )
    final_transcripts = transcript_summary.loc[final_mask].copy()
    final_transcripts = final_transcripts.sort_values(
        ["prevalence_tpm_gt_1", "median_tpm", "mean_tpm"],
        ascending=False,
    )
    LOG.info(
        "Transcript filter applied: median TPM > %.3f and prevalence(TPM>1) >= %.1f%% -> retained_transcripts=%s, genes=%s",
        args.transcript_median_tpm_min,
        args.transcript_prevalence_min * 100,
        f"{len(final_transcripts):,}",
        f"{final_transcripts['gene_id'].nunique():,}",
    )

    final_transcripts_file = args.output_dir / "muscle_selected_lncRNA_transcripts.tsv"
    final_transcripts.to_csv(final_transcripts_file, sep="\t", index=False)
    LOG.info("Saved final transcript list: %s", final_transcripts_file)

    version_map = final_transcripts[
        [
            "transcript_id",
            "transcript_id_versioned",
            "gene_id",
            "gene_id_versioned",
            "gene_name",
            "transcript_name",
            "transcript_type",
            "gene_type",
        ]
    ].drop_duplicates()
    version_map_file = args.output_dir / "muscle_selected_lncRNA_transcripts_version_map.tsv"
    version_map.to_csv(version_map_file, sep="\t", index=False)
    LOG.info("Saved transcript version map: %s", version_map_file)

    output_fasta = args.output_dir / "muscle_selected_lncRNA_transcripts_versioned.fa"
    _, _, found_ids = extract_selected_fasta(
        fasta_path=args.transcripts_fasta,
        selected_versioned_ids=set(final_transcripts["transcript_id_versioned"]),
        output_fasta=output_fasta,
    )

    missing_ids = sorted(set(final_transcripts["transcript_id_versioned"]) - found_ids)
    if missing_ids:
        missing_file = args.output_dir / "muscle_selected_lncRNA_transcripts_missing_in_fasta.txt"
        missing_file.write_text("\n".join(missing_ids) + "\n")
        LOG.warning("Some transcript sequences were missing in FASTA. Saved list: %s", missing_file)

    report_lines = [
        f"Gene TPM file: {args.gene_tpm}",
        f"Transcript TPM file: {args.transcript_tpm}",
        f"Sample attributes: {args.sample_attr if args.sample_attr else 'not provided; inferred from gene TPM columns'}",
        f"GTF file: {args.gtf}",
        f"Transcript FASTA: {args.transcripts_fasta}",
        f"Tissue: {args.tissue}",
        f"Selected biotypes: {', '.join(args.lncrna_biotypes)}",
        f"Gene-level retained genes: {len(filtered_genes)}",
        f"Candidate transcripts from retained genes: {len(candidate_tx)}",
        f"Transcript summary rows after TPM intersection: {len(transcript_summary)}",
        f"Final retained transcripts: {len(final_transcripts)}",
        f"Final retained genes at transcript stage: {final_transcripts['gene_id'].nunique()}",
        f"FASTA sequences written: {len(found_ids)}",
    ]
    report_file = args.output_dir / "preparation_report.txt"
    report_file.write_text("\n".join(report_lines) + "\n")
    LOG.info("Saved report: %s", report_file)

    LOG.info("Done.")

def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_yaml_config(args.config)
    if args.gene_tpm is None:
        args.gene_tpm = resolve_path(get_nested(cfg, "paths", "gene_tpm"), cfg)
    if args.transcript_tpm is None:
        args.transcript_tpm = resolve_path(get_nested(cfg, "paths", "transcript_tpm"), cfg)
    if args.sample_attr is None:
        sample_attr = get_nested(cfg, "paths", "sample_attr")
        args.sample_attr = resolve_path(sample_attr, cfg) if sample_attr else None
    if args.gtf is None:
        args.gtf = resolve_path(get_nested(cfg, "paths", "gtf"), cfg)
    if args.transcripts_fasta is None:
        args.transcripts_fasta = resolve_path(get_nested(cfg, "paths", "transcripts_fasta"), cfg)
    if args.output_dir is None:
        args.output_dir = resolve_path(get_nested(cfg, "stage_outputs", "prepare_dir"), cfg)
    if args.tissue is None:
        args.tissue = get_nested(cfg, "prepare", "tissue", default="Muscle - Skeletal")
    if args.lncrna_biotypes is None:
        args.lncrna_biotypes = list(get_nested(cfg, "prepare", "lncRNA_biotypes", default=["lncRNA"]))
    if args.gene_tpm_threshold is None:
        args.gene_tpm_threshold = float(get_nested(cfg, "prepare", "gene_tpm_threshold", default=1.0))
    if args.gene_min_fraction is None:
        args.gene_min_fraction = float(get_nested(cfg, "prepare", "gene_min_fraction", default=0.10))
    if args.transcript_median_tpm_min is None:
        args.transcript_median_tpm_min = float(get_nested(cfg, "prepare", "transcript_median_tpm_min", default=0.5))
    if args.transcript_prevalence_min is None:
        args.transcript_prevalence_min = float(get_nested(cfg, "prepare", "transcript_prevalence_min", default=0.10))
    if args.chunksize is None:
        args.chunksize = int(get_nested(cfg, "prepare", "chunksize", default=5000))

    required = ["gene_tpm", "transcript_tpm", "gtf", "transcripts_fasta", "output_dir"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments after config resolution: {missing}")
    return args

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare skeletal muscle lncRNA genes, transcripts, and FASTA from GTEx + GENCODE.")
    parser.add_argument("--config", type=Path, default=None, help="YAML config file with pipeline paths and parameters.")
    parser.add_argument("--gene-tpm", type=Path, default=None, help="GTEx gene-level TPM GCT file.")
    parser.add_argument("--transcript-tpm", type=Path, default=None, help="GTEx transcript-level TPM table.")
    parser.add_argument("--sample-attr", type=Path, default=None, help="GTEx sample attributes table.")
    parser.add_argument("--gtf", type=Path, default=None, help="GENCODE GTF annotation file.")
    parser.add_argument("--transcripts-fasta", type=Path, default=None, help="GENCODE transcript FASTA file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for all outputs.")
    parser.add_argument("--tissue", type=str, default=None, help="Target GTEx tissue name.")
    parser.add_argument("--lncrna-biotypes", nargs="+", default=None, help="GENCODE gene/transcript biotypes to retain.")
    parser.add_argument("--gene-tpm-threshold", type=float, default=None, help="Gene-level TPM threshold. Allowed: 0.1, 0.5, 1.0")
    parser.add_argument("--gene-min-fraction", type=float, default=None, help="Minimum gene-level sample fraction for selected TPM threshold.")
    parser.add_argument("--transcript-median-tpm-min", type=float, default=None, help="Minimum transcript median TPM.")
    parser.add_argument("--transcript-prevalence-min", type=float, default=None, help="Minimum transcript prevalence for TPM>1.")
    parser.add_argument("--chunksize", type=int, default=None, help="Chunk size for transcript TPM reading.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args = apply_config_defaults(args)
    setup_logging(args.verbose)
    prepare_data(args)

if __name__ == "__main__":
    main()
