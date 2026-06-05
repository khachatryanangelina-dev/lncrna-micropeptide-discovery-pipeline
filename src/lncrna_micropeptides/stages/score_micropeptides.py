#!/usr/bin/env python3
from __future__ import annotations

"""
Score stop-to-stop nucleotide ORF candidates after ORF annotation.

This stage is designed for the nucleotide-first / stop-to-stop pipeline:
- input: 07_annotated_orf_candidates.tsv
- output: ranked feature table + top candidate tables

Design principle:
- ORFs are not treated as experimentally validated coding sequences.
- Scores are continuous prioritization features, not binary validation.
- Peptide-derived features are secondary support computed from in-frame translation.
"""

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from lncrna_micropeptides.pipeline_config import get_nested, load_yaml_config, resolve_path

# Logging / small utilities

def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def to_num(s: pd.Series | object) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.Series([s]).pipe(pd.to_numeric, errors="coerce")


def bool_score(series: pd.Series, default: float = np.nan) -> pd.Series:
    def convert(x: object) -> float:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            x2 = x.strip().lower()
            if x2 in {"true", "1", "yes", "y"}:
                return 1.0
            if x2 in {"false", "0", "no", "n"}:
                return 0.0
        return float(bool(x))

    return series.map(convert).astype(float)


def clipped_linear(x: object, low: float, high: float) -> float:
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(x):
        return np.nan
    if high <= low:
        return float(x >= high)
    return float(np.clip((x - low) / (high - low), 0.0, 1.0))


def inverse_clipped_linear(x: object, low: float, high: float) -> float:
    val = clipped_linear(x, low, high)
    return np.nan if pd.isna(val) else 1.0 - val


def trapezoid_preference(x: object, low_zero: float, low_one: float, high_one: float, high_zero: float) -> float:
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(x):
        return np.nan
    if x <= low_zero or x >= high_zero:
        return 0.0
    if low_one <= x <= high_one:
        return 1.0
    if x < low_one:
        return float((x - low_zero) / (low_one - low_zero))
    return float((high_zero - x) / (high_zero - high_one))


def safe_log1p_score(x: object, low: float, high: float) -> float:
    x = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(x):
        return np.nan
    return clipped_linear(math.log1p(max(x, 0.0)), math.log1p(low), math.log1p(high))


def rowwise_weighted_mean(df: pd.DataFrame, components: list[tuple[str, float]]) -> pd.Series:
    vals = []
    for _, row in df.iterrows():
        xs = []
        ws = []
        for col, weight in components:
            if col not in row.index:
                continue
            v = row[col]
            if pd.notna(v):
                xs.append(float(v))
                ws.append(float(weight))
        vals.append(np.nan if not ws else float(np.average(xs, weights=ws)))
    return pd.Series(vals, index=df.index, dtype=float)


def get_first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

# Loading

def load_annotated_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    required = {"transcript_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input table lacks required columns: {sorted(missing)}")
    return df

# Score components

def add_qc_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bool_components = {
        "score_sequence_reconstructed": "sequence_reconstructed_successfully",
        "score_coordinates_valid": "orf_coordinates_valid",
        "score_within_bounds": "orf_within_transcript_bounds",
        "score_length_matches_span": get_first_existing_col(df, ["length_nt_matches_span", "nt_length_matches_span"]),
        "score_mod3": get_first_existing_col(df, ["length_mod3_ok"]),
    }
    for out_col, in_col in bool_components.items():
        if in_col and in_col in df.columns:
            df[out_col] = bool_score(df[in_col])
        else:
            df[out_col] = np.nan

    if "contains_internal_stop" in df.columns:
        df["score_no_internal_stop"] = 1.0 - bool_score(df["contains_internal_stop"], default=0.0)
    else:
        df["score_no_internal_stop"] = np.nan

    if "passes_min_nt_filter" in df.columns:
        df["score_passes_min_nt"] = bool_score(df["passes_min_nt_filter"])
    else:
        df["score_passes_min_nt"] = np.nan

    # terminal_triplet_is_stop is deliberately not required: getorf -find 2 often returns regions without including stop codons.
    df["qc_integrity_score"] = rowwise_weighted_mean(
        df,
        [
            ("score_sequence_reconstructed", 0.20),
            ("score_coordinates_valid", 0.15),
            ("score_within_bounds", 0.15),
            ("score_length_matches_span", 0.15),
            ("score_mod3", 0.15),
            ("score_no_internal_stop", 0.15),
            ("score_passes_min_nt", 0.05),
        ],
    )
    return df


def add_structural_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    length_col = get_first_existing_col(df, ["length_nt", "orf_length_nt"])
    aa_col = get_first_existing_col(df, ["translated_peptide_length", "peptide_length", "length_aa"])

    if length_col:
        length_nt = to_num(df[length_col])
        df["score_length_nt_window"] = length_nt.map(lambda x: trapezoid_preference(x, 60, 90, 300, 450))
        df["score_length_nt_longer_than_min"] = length_nt.map(lambda x: clipped_linear(x, 60, 180))
    else:
        df["score_length_nt_window"] = np.nan
        df["score_length_nt_longer_than_min"] = np.nan

    if aa_col:
        length_aa = to_num(df[aa_col])
        df["score_peptide_length_window"] = length_aa.map(lambda x: trapezoid_preference(x, 20, 30, 100, 150))
    else:
        df["score_peptide_length_window"] = np.nan

    if "gc_content" in df.columns:
        df["score_gc_window"] = to_num(df["gc_content"]).map(lambda x: trapezoid_preference(x, 0.25, 0.35, 0.65, 0.80))
    else:
        df["score_gc_window"] = np.nan

    if "gc3_content" in df.columns:
        df["score_gc3_window"] = to_num(df["gc3_content"]).map(lambda x: trapezoid_preference(x, 0.20, 0.35, 0.70, 0.85))
    else:
        df["score_gc3_window"] = np.nan

    if "nucleotide_complexity" in df.columns:
        df["score_nucleotide_complexity"] = to_num(df["nucleotide_complexity"]).map(lambda x: clipped_linear(x, 0.75, 0.95))
    else:
        df["score_nucleotide_complexity"] = np.nan

    if "codon_usage_entropy" in df.columns:
        df["score_codon_entropy"] = to_num(df["codon_usage_entropy"]).map(lambda x: clipped_linear(x, 0.45, 0.80))
    else:
        df["score_codon_entropy"] = np.nan

    if "longest_homopolymer_run_nt" in df.columns:
        df["score_no_long_homopolymer"] = to_num(df["longest_homopolymer_run_nt"]).map(lambda x: inverse_clipped_linear(x, 6, 12))
    else:
        df["score_no_long_homopolymer"] = np.nan

    if "cpg_dinucleotide_fraction" in df.columns:
        df["score_cpg_not_extreme"] = to_num(df["cpg_dinucleotide_fraction"]).map(lambda x: trapezoid_preference(x, 0.0, 0.005, 0.08, 0.16))
    else:
        df["score_cpg_not_extreme"] = np.nan

    # Use previous structural score as a helper, but do not let it dominate.
    if "structural_orf_score" in df.columns:
        df["score_previous_structural"] = to_num(df["structural_orf_score"]).clip(0, 1)
    else:
        df["score_previous_structural"] = np.nan

    df["sequence_structure_score"] = rowwise_weighted_mean(
        df,
        [
            ("score_length_nt_window", 0.20),
            ("score_peptide_length_window", 0.10),
            ("score_gc_window", 0.15),
            ("score_gc3_window", 0.10),
            ("score_nucleotide_complexity", 0.15),
            ("score_codon_entropy", 0.15),
            ("score_no_long_homopolymer", 0.08),
            ("score_cpg_not_extreme", 0.04),
            ("score_previous_structural", 0.03),
        ],
    )
    return df


def add_start_context_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # This is weak support, not a hard requirement for stop-to-stop candidates.
    if "start_codon_class" in df.columns:
        start_map = {"ATG": 1.0, "near_cognate": 0.65, "noncanonical": 0.15}
        df["score_start_codon"] = df["start_codon_class"].map(start_map).astype(float)
    elif "first_triplet_class" in df.columns:
        start_map = {"canonical_start": 1.0, "near_cognate_start": 0.65, "other": 0.15, "stop_codon": 0.0}
        df["score_start_codon"] = df["first_triplet_class"].map(start_map).astype(float)
    elif "canonical_start" in df.columns:
        df["score_start_codon"] = bool_score(df["canonical_start"])
    else:
        df["score_start_codon"] = np.nan

    if "kozak_score" in df.columns:
        df["score_kozak"] = to_num(df["kozak_score"]).clip(0, 1)
    else:
        df["score_kozak"] = np.nan

    if "kozak_strength" in df.columns:
        df["score_kozak_strength"] = df["kozak_strength"].map({"strong": 1.0, "moderate": 0.65, "weak": 0.25, "unknown": np.nan}).astype(float)
    else:
        df["score_kozak_strength"] = np.nan

    if "has_purine_at_minus3" in df.columns:
        df["score_minus3_purine"] = bool_score(df["has_purine_at_minus3"])
    else:
        df["score_minus3_purine"] = np.nan

    if "has_G_at_plus4" in df.columns:
        df["score_plus4_g"] = bool_score(df["has_G_at_plus4"])
    else:
        df["score_plus4_g"] = np.nan

    df["translation_start_context_score"] = rowwise_weighted_mean(
        df,
        [
            ("score_start_codon", 0.40),
            ("score_kozak", 0.25),
            ("score_kozak_strength", 0.15),
            ("score_minus3_purine", 0.10),
            ("score_plus4_g", 0.10),
        ],
    )
    return df


def add_transcript_context_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "is_representative_orf" in df.columns:
        df["score_representative"] = bool_score(df["is_representative_orf"])
    else:
        df["score_representative"] = np.nan

    longest_col = get_first_existing_col(df, ["is_longest_orf_in_tx", "is_longest_orf_in_transcript"])
    if longest_col:
        df["score_longest_in_tx"] = bool_score(df[longest_col])
    else:
        df["score_longest_in_tx"] = np.nan

    rel_len_col = get_first_existing_col(df, ["length_nt_relative_to_longest_in_tx", "length_aa_relative_to_longest_in_tx"])
    if rel_len_col:
        df["score_relative_to_longest"] = to_num(df[rel_len_col]).clip(0, 1)
    else:
        df["score_relative_to_longest"] = np.nan

    rank_col = get_first_existing_col(df, ["orf_rank_by_length_in_tx", "length_rank_within_transcript"])
    if rank_col:
        rank = to_num(df[rank_col])
        df["score_rank_in_tx"] = rank.map(lambda x: 1.0 / x if pd.notna(x) and x > 0 else np.nan)
    else:
        df["score_rank_in_tx"] = np.nan

    burden_col = get_first_existing_col(df, ["orfs_per_transcript", "n_orfs_in_transcript"])
    if burden_col:
        burden = to_num(df[burden_col])
        # Penalize heavily fragmented transcripts, but do not zero them out.
        df["score_low_orf_burden"] = burden.map(lambda x: 1.0 / math.sqrt(x) if pd.notna(x) and x > 0 else np.nan)
        df["score_low_orf_burden"] = df["score_low_orf_burden"].map(lambda x: clipped_linear(x, 0.05, 0.50))
    else:
        df["score_low_orf_burden"] = np.nan

    if "orf_fraction_of_transcript" in df.columns:
        df["score_orf_fraction"] = to_num(df["orf_fraction_of_transcript"]).map(lambda x: trapezoid_preference(x, 0.01, 0.03, 0.35, 0.75))
    else:
        df["score_orf_fraction"] = np.nan

    region_col = get_first_existing_col(df, ["orf_region"])
    if region_col:
        df["score_region"] = df[region_col].map({"5prime": 1.0, "middle": 0.75, "3prime": 0.45}).astype(float)
    else:
        df["score_region"] = np.nan

    df["transcript_context_score"] = rowwise_weighted_mean(
        df,
        [
            ("score_representative", 0.25),
            ("score_longest_in_tx", 0.20),
            ("score_relative_to_longest", 0.15),
            ("score_rank_in_tx", 0.15),
            ("score_low_orf_burden", 0.10),
            ("score_orf_fraction", 0.10),
            ("score_region", 0.05),
        ],
    )
    return df


def add_overlap_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "max_overlap_fraction_with_any_orf" in df.columns:
        ov = to_num(df["max_overlap_fraction_with_any_orf"])
        # High overlap is not fatal, but it weakens independence of a candidate.
        df["score_low_overlap"] = ov.map(lambda x: inverse_clipped_linear(x, 0.40, 0.95))
    else:
        df["score_low_overlap"] = np.nan

    if "nested_within_another_orf" in df.columns:
        df["score_not_nested"] = 1.0 - bool_score(df["nested_within_another_orf"], default=0.0)
    else:
        df["score_not_nested"] = np.nan

    if "contains_another_orf" in df.columns:
        df["score_contains_other_orf"] = bool_score(df["contains_another_orf"], default=0.0)
    else:
        df["score_contains_other_orf"] = np.nan

    df["overlap_independence_score"] = rowwise_weighted_mean(
        df,
        [
            ("score_low_overlap", 0.40),
            ("score_not_nested", 0.40),
            ("score_contains_other_orf", 0.20),
        ],
    )
    return df


def add_expression_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    median_col = get_first_existing_col(df, ["median_tpm", "median_TPM"])
    mean_col = get_first_existing_col(df, ["mean_tpm", "mean_TPM"])
    max_col = get_first_existing_col(df, ["max_tpm", "max_TPM"])
    prev_col = get_first_existing_col(df, ["prevalence_tpm_gt_1", "prevalence_gt_1", "fraction_samples_tpm_gt_1"])
    prev05_col = get_first_existing_col(df, ["prevalence_tpm_gt_0_5"])

    df["score_median_tpm"] = to_num(df[median_col]).map(lambda x: safe_log1p_score(x, 0.3, 10.0)) if median_col else np.nan
    df["score_mean_tpm"] = to_num(df[mean_col]).map(lambda x: safe_log1p_score(x, 0.3, 15.0)) if mean_col else np.nan
    df["score_max_tpm"] = to_num(df[max_col]).map(lambda x: safe_log1p_score(x, 1.0, 50.0)) if max_col else np.nan
    df["score_prevalence_gt1"] = to_num(df[prev_col]).map(lambda x: clipped_linear(x, 0.03, 0.50)) if prev_col else np.nan
    df["score_prevalence_gt05"] = to_num(df[prev05_col]).map(lambda x: clipped_linear(x, 0.05, 0.60)) if prev05_col else np.nan

    df["expression_support_score"] = rowwise_weighted_mean(
        df,
        [
            ("score_median_tpm", 0.35),
            ("score_mean_tpm", 0.20),
            ("score_max_tpm", 0.10),
            ("score_prevalence_gt1", 0.25),
            ("score_prevalence_gt05", 0.10),
        ],
    )
    return df


def add_peptide_support_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # These are descriptive supports, not proof of translation.
    gravy_col = get_first_existing_col(df, ["translated_gravy", "gravy"])
    charge_col = get_first_existing_col(df, ["translated_net_charge_pH7", "net_charge_pH7"])
    pi_col = get_first_existing_col(df, ["translated_isoelectric_point", "isoelectric_point"])

    if gravy_col:
        df["score_gravy_window"] = to_num(df[gravy_col]).map(lambda x: trapezoid_preference(x, -2.5, -0.8, 1.2, 2.8))
    else:
        df["score_gravy_window"] = np.nan

    if charge_col:
        df["score_charge_window"] = to_num(df[charge_col]).map(lambda x: trapezoid_preference(abs(x) if pd.notna(x) else np.nan, 0.0, 0.5, 5.0, 10.0))
        df["score_basic_tendency"] = to_num(df[charge_col]).map(lambda x: clipped_linear(x, 0.0, 4.0))
    else:
        df["score_charge_window"] = np.nan
        df["score_basic_tendency"] = np.nan

    if pi_col:
        df["score_pi_window"] = to_num(df[pi_col]).map(lambda x: trapezoid_preference(x, 4.0, 5.5, 11.0, 13.0))
    else:
        df["score_pi_window"] = np.nan

    for out_col, in_col in [
        ("score_tm_like", "tm_helix_like"),
        ("score_signal_like", "signal_like_nterm"),
        ("score_positive_tail", "positive_tail_motif"),
    ]:
        df[out_col] = bool_score(df[in_col], default=0.0) if in_col in df.columns else np.nan

    archetype_cols = [c for c in ["score_tm_like", "score_signal_like", "score_positive_tail", "score_basic_tendency"] if c in df.columns]
    df["score_archetype_support"] = df[archetype_cols].max(axis=1, skipna=True) if archetype_cols else np.nan

    if "low_complexity_flag" in df.columns:
        df["score_no_peptide_low_complexity"] = 1.0 - bool_score(df["low_complexity_flag"], default=0.0)
    else:
        df["score_no_peptide_low_complexity"] = np.nan

    for frac_col in ["fraction_hydrophobic_residues", "fraction_charged_residues", "fraction_polar_residues", "aromaticity"]:
        if frac_col not in df.columns:
            df[frac_col] = np.nan

    df["score_hydrophobic_fraction"] = to_num(df["fraction_hydrophobic_residues"]).map(lambda x: trapezoid_preference(x, 0.15, 0.25, 0.65, 0.85))
    df["score_charged_fraction"] = to_num(df["fraction_charged_residues"]).map(lambda x: trapezoid_preference(x, 0.02, 0.06, 0.35, 0.60))
    df["score_polar_fraction"] = to_num(df["fraction_polar_residues"]).map(lambda x: trapezoid_preference(x, 0.02, 0.08, 0.45, 0.70))
    df["score_aromaticity"] = to_num(df["aromaticity"]).map(lambda x: trapezoid_preference(x, 0.0, 0.02, 0.20, 0.40))

    df["translated_peptide_support_score"] = rowwise_weighted_mean(
        df,
        [
            ("score_gravy_window", 0.18),
            ("score_charge_window", 0.12),
            ("score_pi_window", 0.08),
            ("score_archetype_support", 0.20),
            ("score_no_peptide_low_complexity", 0.12),
            ("score_hydrophobic_fraction", 0.10),
            ("score_charged_fraction", 0.08),
            ("score_polar_fraction", 0.06),
            ("score_aromaticity", 0.06),
        ],
    )
    return df


def add_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["integrated_micropeptide_score"] = rowwise_weighted_mean(
        df,
        [
            ("qc_integrity_score", 0.12),
            ("sequence_structure_score", 0.20),
            ("translation_start_context_score", 0.12),
            ("transcript_context_score", 0.22),
            ("overlap_independence_score", 0.12),
            ("expression_support_score", 0.12),
            ("translated_peptide_support_score", 0.10),
        ],
    )

    df["score_percentile"] = df["integrated_micropeptide_score"].rank(pct=True, method="average")

    conditions = [
        df["integrated_micropeptide_score"] >= 0.75,
        df["integrated_micropeptide_score"] >= 0.55,
    ]
    choices = ["high", "medium"]
    df["score_confidence"] = np.select(conditions, choices, default="low")

    sort_cols = [
        "integrated_micropeptide_score",
        "transcript_context_score",
        "sequence_structure_score",
        "translation_start_context_score",
        "expression_support_score",
    ]
    existing_sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(existing_sort_cols, ascending=[False] * len(existing_sort_cols)).reset_index(drop=True)
    df["global_rank"] = np.arange(1, len(df) + 1)
    if "transcript_id" in df.columns:
        df["rank_within_transcript_by_score"] = df.groupby("transcript_id")["integrated_micropeptide_score"].rank(method="first", ascending=False).astype("Int64")
        df["is_top_scored_orf_in_transcript"] = df["rank_within_transcript_by_score"].eq(1).astype("boolean")
    return df


def build_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = add_qc_scores(df)
    df = add_structural_scores(df)
    df = add_start_context_scores(df)
    df = add_transcript_context_scores(df)
    df = add_overlap_scores(df)
    df = add_expression_scores(df)
    df = add_peptide_support_scores(df)
    df = add_final_scores(df)
    return df

# Outputs 

def coalesce_first_nonempty(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return first non-empty value across candidate columns for each row."""
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in candidates:
        if col not in df.columns:
            continue
        vals = df[col].astype("object")
        vals = vals.where(~vals.astype(str).str.lower().isin(["", "nan", "none", "<na>"]), pd.NA)
        out = out.where(out.notna(), vals)
    return out


def add_gene_level_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable gene-level keys for deduplication.

    The annotated table can contain gene_id/gene_name columns from several merge steps
    (for example gene_id_x/gene_id_y/gene_id). This function creates unified columns
    used only for ranking/deduplication, without deleting original columns.
    """
    out = df.copy()
    out["gene_id_unified"] = coalesce_first_nonempty(out, ["gene_id", "gene_id_x", "gene_id_y"])
    out["gene_name_unified"] = coalesce_first_nonempty(out, ["gene_name", "gene_name_x", "gene_name_y"])
    if "transcript_id" in out.columns:
        out["gene_level_key"] = out["gene_id_unified"].fillna(out["transcript_id"])
    else:
        out["gene_level_key"] = out["gene_id_unified"].fillna(out.index.astype(str))
    return out


def make_top_tables(df: pd.DataFrame, top_n: int, top_source: str) -> dict[str, pd.DataFrame]:
    df = add_gene_level_keys(df)
    tables: dict[str, pd.DataFrame] = {}
    tables["top_all"] = df.head(min(top_n, len(df))).copy()

    if "is_representative_orf" in df.columns:
        rep = df[df["is_representative_orf"].fillna(False).astype(bool)].copy()
    else:
        rep = pd.DataFrame(columns=df.columns)
    tables["representative"] = rep
    tables["top_representative"] = rep.head(min(top_n, len(rep))).copy()

    if "score_confidence" in df.columns:
        high_medium = df[df["score_confidence"].isin(["high", "medium"])].copy()
        high = df[df["score_confidence"].eq("high")].copy()
    else:
        high_medium = pd.DataFrame(columns=df.columns)
        high = pd.DataFrame(columns=df.columns)
    tables["high_confidence"] = high
    tables["top_high_or_medium"] = high_medium.head(min(top_n, len(high_medium))).copy()

    # One best candidate per transcript, then top N transcripts.
    if "rank_within_transcript_by_score" in df.columns:
        per_tx = df[df["rank_within_transcript_by_score"].eq(1)].copy()
    else:
        per_tx = df.drop_duplicates("transcript_id").copy() if "transcript_id" in df.columns else pd.DataFrame(columns=df.columns)
    tables["best_per_transcript"] = per_tx
    tables["top_best_per_transcript"] = per_tx.head(min(top_n, len(per_tx))).copy()

    if "gene_level_key" in df.columns:
        per_gene = df.drop_duplicates("gene_level_key", keep="first").copy()
    else:
        per_gene = pd.DataFrame(columns=df.columns)
    tables["best_per_gene"] = per_gene
    tables["top_best_per_gene"] = per_gene.head(min(top_n, len(per_gene))).copy()

    if top_source == "all":
        tables["selected_top"] = tables["top_all"]
    elif top_source == "representative":
        tables["selected_top"] = tables["top_representative"]
    elif top_source == "high_or_medium":
        tables["selected_top"] = tables["top_high_or_medium"]
    elif top_source == "best_per_transcript":
        tables["selected_top"] = tables["top_best_per_transcript"]
    elif top_source == "best_per_gene":
        tables["selected_top"] = tables["top_best_per_gene"]
    else:
        raise ValueError(f"Unsupported top_source: {top_source}")
    return tables


def summarize(df: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> str:
    lines = [
        f"rows\t{len(df)}",
        f"transcripts\t{df['transcript_id'].nunique() if 'transcript_id' in df.columns else 'NA'}",
        f"columns\t{len(df.columns)}",
    ]

    for col in [
        "qc_integrity_score",
        "sequence_structure_score",
        "translation_start_context_score",
        "transcript_context_score",
        "overlap_independence_score",
        "expression_support_score",
        "translated_peptide_support_score",
        "integrated_micropeptide_score",
    ]:
        if col in df.columns:
            x = to_num(df[col])
            lines.extend([
                f"{col}_median\t{x.median()}",
                f"{col}_mean\t{x.mean()}",
                f"{col}_max\t{x.max()}",
            ])

    if "score_confidence" in df.columns:
        counts = df["score_confidence"].value_counts(dropna=False).to_dict()
        for label in ["high", "medium", "low"]:
            lines.append(f"score_confidence_{label}\t{counts.get(label, 0)}")

    for name, tab in tables.items():
        lines.append(f"{name}_rows\t{len(tab)}")

    return "\n".join(map(str, lines)) + "\n"


def choose_compact_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "global_rank",
        "score_percentile",
        "integrated_micropeptide_score",
        "score_confidence",
        "transcript_id",
        "gene_level_key", "gene_id_unified", "gene_name_unified",
        "gene_id_x", "gene_id_y", "gene_id",
        "gene_name_x", "gene_name_y", "gene_name",
        "transcript_biotype", "transcript_type",
        "raw_orf_id",
        "orf_index",
        "start_1based", "end_1based", "orf_start_nt", "orf_end_nt",
        "length_nt", "orf_length_nt", "length_aa", "translated_peptide_length", "peptide_length",
        "nucleotide_sequence",
        "translated_from_nt", "peptide_sequence_clean", "peptide_sequence",
        "first_triplet", "first_triplet_class", "start_codon", "start_codon_class",
        "kozak_score", "kozak_strength",
        "gc_content", "gc3_content", "nucleotide_complexity", "codon_usage_entropy",
        "orf_fraction_of_transcript", "orf_region",
        "orfs_per_transcript", "n_orfs_in_transcript",
        "is_representative_orf", "representative_orf_rank",
        "is_longest_orf_in_tx", "is_longest_orf_in_transcript",
        "rank_within_transcript_by_score", "is_top_scored_orf_in_transcript",
        "max_overlap_fraction_with_any_orf", "nested_within_another_orf", "contains_another_orf",
        "median_tpm", "mean_tpm", "max_tpm", "prevalence_tpm_gt_1", "prevalence_tpm_gt_0_5",
        "qc_integrity_score", "sequence_structure_score", "translation_start_context_score",
        "transcript_context_score", "overlap_independence_score", "expression_support_score",
        "translated_peptide_support_score",
        "gravy", "translated_gravy", "net_charge_pH7", "translated_net_charge_pH7",
        "isoelectric_point", "translated_isoelectric_point",
        "tm_helix_like", "signal_like_nterm", "positive_tail_motif", "dominant_archetype",
    ]
    seen = set()
    keep = []
    for c in preferred:
        if c in df.columns and c not in seen:
            keep.append(c)
            seen.add(c)
    return keep


def save_outputs(df: pd.DataFrame, output_dir: Path, top_n: int, top_source: str, verbose: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = make_top_tables(df, top_n=top_n, top_source=top_source)
    compact_cols = choose_compact_columns(df)

    full_path = output_dir / "08_micropeptide_scored_feature_table.tsv"
    compact_path = output_dir / "08_micropeptide_scored_feature_table_compact.tsv"
    selected_top_path = output_dir / f"08_top_{top_n}_micropeptide_candidates.tsv"
    top50_path = output_dir / "08_top_50_micropeptide_candidates.tsv"
    top100_path = output_dir / "08_top_100_micropeptide_candidates.tsv"
    rep_path = output_dir / "08_representative_orf_table.tsv"
    best_tx_path = output_dir / "08_best_orf_per_transcript.tsv"
    best_gene_path = output_dir / "08_best_orf_per_gene.tsv"
    top50_gene_path = output_dir / "08_top_50_gene_level_micropeptide_candidates.tsv"
    top100_gene_path = output_dir / "08_top_100_gene_level_micropeptide_candidates.tsv"
    summary_path = output_dir / "08_micropeptide_scoring_summary.txt"

    df.to_csv(full_path, sep="\t", index=False)
    df[compact_cols].to_csv(compact_path, sep="\t", index=False)
    tables["selected_top"][compact_cols].to_csv(selected_top_path, sep="\t", index=False)
    df.head(min(50, len(df)))[compact_cols].to_csv(top50_path, sep="\t", index=False)
    df.head(min(100, len(df)))[compact_cols].to_csv(top100_path, sep="\t", index=False)
    tables["representative"][compact_cols].to_csv(rep_path, sep="\t", index=False)
    tables["best_per_transcript"][compact_cols].to_csv(best_tx_path, sep="\t", index=False)
    tables["best_per_gene"][compact_cols].to_csv(best_gene_path, sep="\t", index=False)
    tables["best_per_gene"].head(min(50, len(tables["best_per_gene"])))[compact_cols].to_csv(top50_gene_path, sep="\t", index=False)
    tables["best_per_gene"].head(min(100, len(tables["best_per_gene"])))[compact_cols].to_csv(top100_gene_path, sep="\t", index=False)
    summary_path.write_text(summarize(df, tables), encoding="utf-8")

    try:
        df.to_parquet(output_dir / "08_micropeptide_scored_feature_table.parquet", index=False)
    except Exception as e:
        warn(f"Parquet export skipped: {e}")

    log(f"Saved full scored table: {full_path}", verbose)
    log(f"Saved compact scored table: {compact_path}", verbose)
    log(f"Saved selected top table: {selected_top_path}", verbose)
    log(f"Saved summary: {summary_path}", verbose)


# CLI / config

def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_yaml_config(args.config)

    if args.input_tsv is None:
        annotate_dir = get_nested(cfg, "stage_outputs", "annotate_dir")
        if annotate_dir:
            args.input_tsv = resolve_path(Path(annotate_dir) / "07_annotated_orf_candidates.tsv", cfg)

    if args.output_dir is None:
        score_dir = get_nested(cfg, "stage_outputs", "score_dir")
        if score_dir:
            args.output_dir = resolve_path(score_dir, cfg)

    if args.top_n is None:
        args.top_n = int(get_nested(cfg, "score", "top_n", default=100))

    if args.top_source is None:
        args.top_source = str(get_nested(cfg, "score", "top_source", default="best_per_transcript"))

    if args.input_tsv is None or args.output_dir is None:
        raise ValueError("input-tsv and output-dir must be provided directly or through config")

    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score stop-to-stop lncRNA ORF candidates and export top-ranked micropeptide candidates.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--input-tsv", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--top-n", type=int, default=None)
    p.add_argument(
        "--top-source",
        choices=["all", "representative", "high_or_medium", "best_per_transcript", "best_per_gene"],
        default=None,
        help="Which candidate subset should be used for the main top-N table.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = apply_config_defaults(parse_args())
    input_tsv = Path(args.input_tsv)
    output_dir = Path(args.output_dir)

    log(f"Loading annotated ORF table: {input_tsv}", args.verbose)
    df = load_annotated_table(input_tsv)
    log(f"Loaded rows: {len(df):,}", args.verbose)

    scored = build_scores(df)
    scored = add_gene_level_keys(scored)
    save_outputs(scored, output_dir=output_dir, top_n=args.top_n, top_source=args.top_source, verbose=args.verbose)
    log("Done.", args.verbose)


if __name__ == "__main__":
    main()
