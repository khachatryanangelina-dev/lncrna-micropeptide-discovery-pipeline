#!/usr/bin/env python3
from __future__ import annotations

"""
Annotate stop-stop nucleotide ORF candidates.

This version is aligned with a nucleotide-first, stop-stop GETORF stage.
It deliberately avoids peptide-centric confidence logic as a primary decision rule.

Core principles:
- treat ORFs as stop-delimited nucleotide segments, not validated proteins;
- compute structural, positional, compositional, and transcript-context features;
- optionally derive translated peptide-like descriptors only as secondary descriptors;
- build a transparent structural prioritization score instead of hard-coded
  start-stop / peptide-validation confidence classes.
"""

import argparse
import gzip
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lncrna_micropeptides.pipeline_config import get_nested, load_yaml_config, resolve_path

STOP_CODONS = {"TAA", "TAG", "TGA"}
NEAR_COGNATE_STARTS = {"CTG", "GTG", "TTG", "ACG", "ATA", "ATC", "ATT"}

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

HYDROPHOBIC_AA = set("AVILMFWYC")
CHARGED_AA = set("DEKRH")
POSITIVE_AA = set("KRH")
NEGATIVE_AA = set("DE")
AA_AROMATIC = set("FWYH")
KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
AA_MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.15, "E": 147.13,
    "Q": 146.15, "G": 75.07, "H": 155.16, "I": 131.17, "L": 131.17, "K": 146.19,
    "M": 149.21, "F": 165.19, "P": 115.13, "S": 105.09, "T": 119.12, "W": 204.23,
    "Y": 181.19, "V": 117.15,
}
PKA_SIDE = {"D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1, "H": 6.0, "K": 10.5, "R": 12.5}
PKA_NTERM = 9.69
PKA_CTERM = 2.34

SAMPLE_ID_PATTERNS = [
    re.compile(r"^GTEX-[A-Z0-9\-]+$"),
    re.compile(r"^[A-Z0-9]+-[A-Z0-9]+-SM-[A-Z0-9]+$"),
]


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def strip_version(x: object) -> object:
    if pd.isna(x):
        return x
    return str(x).split(".", 1)[0]


def clean_nt(seq: object) -> str:
    seq = str(seq).upper().replace("U", "T")
    return re.sub(r"[^ACGTN]", "", seq)


def clean_peptide(seq: object) -> str:
    seq = str(seq).upper().replace("*", "")
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq)


def safe_div(n: float, d: float, default: float = np.nan) -> float:
    return default if d == 0 or pd.isna(d) else n / d


def bool_mean_or_na(series: pd.Series):
    x = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    return pd.NA if x.empty else float(x.mean())


def normalized_entropy(values: Sequence[str], alphabet_size: int) -> float:
    if not values or alphabet_size <= 1:
        return np.nan
    counts = Counter(values)
    probs = np.array(list(counts.values()), dtype=float)
    probs /= probs.sum()
    ent = float(-(probs * np.log2(probs)).sum())
    return float(ent / np.log2(alphabet_size))


def split_codons(seq: str) -> List[str]:
    seq = clean_nt(seq)
    usable = len(seq) - (len(seq) % 3)
    return [seq[i:i + 3] for i in range(0, usable, 3)]


def coding_codons_excluding_terminal_stop(seq: str) -> List[str]:
    codons = split_codons(seq)
    if codons and codons[-1] in STOP_CODONS:
        codons = codons[:-1]
    return [c for c in codons if c not in STOP_CODONS]


def translate_nt(seq: str, stop_at_first_stop: bool = True) -> str:
    nt = clean_nt(seq)
    aa = []
    usable = len(nt) - (len(nt) % 3)
    for i in range(0, usable, 3):
        codon = nt[i:i + 3]
        residue = CODON_TABLE.get(codon, "X")
        if residue == "*" and stop_at_first_stop:
            break
        aa.append(residue)
    return "".join(aa)


def count_internal_stops(seq: str) -> int:
    codons = split_codons(seq)
    if not codons:
        return 0
    internal = codons[:-1] if codons[-1] in STOP_CODONS else codons
    return sum(c in STOP_CODONS for c in internal)


def gc_content(seq: str) -> float:
    nt = clean_nt(seq)
    valid = sum(b in {"A", "C", "G", "T"} for b in nt)
    if valid == 0:
        return np.nan
    gc = sum(b in {"G", "C"} for b in nt)
    return gc / valid


def gc3_content(seq: str) -> float:
    codons = coding_codons_excluding_terminal_stop(seq)
    if not codons:
        return np.nan
    third = [c[2] for c in codons]
    return safe_div(sum(nt in {"G", "C"} for nt in third), len(third), np.nan)


def nucleotide_complexity(seq: str) -> float:
    nt = clean_nt(seq)
    return np.nan if not nt else normalized_entropy(list(nt), 4)


def dinucleotide_fraction(seq: str, dinuc: str) -> float:
    nt = clean_nt(seq)
    if len(nt) < 2:
        return np.nan
    kmers = [nt[i:i + 2] for i in range(len(nt) - 1)]
    return safe_div(sum(k == dinuc for k in kmers), len(kmers), np.nan)


def longest_homopolymer_run(seq: str) -> int:
    nt = clean_nt(seq)
    if not nt:
        return 0
    best = cur = 1
    for i in range(1, len(nt)):
        if nt[i] == nt[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def codon_usage_entropy(seq: str) -> float:
    codons = coding_codons_excluding_terminal_stop(seq)
    return normalized_entropy(codons, 64) if codons else np.nan


def classify_first_triplet(codon: object) -> object:
    if pd.isna(codon):
        return pd.NA
    codon = str(codon).upper()
    if codon == "ATG":
        return "canonical_start_like"
    if codon in NEAR_COGNATE_STARTS:
        return "near_cognate_like"
    if codon in STOP_CODONS:
        return "stop"
    return "other"


def robust_minmax(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    lo = x.min()
    hi = x.max()
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(np.nan, index=series.index, dtype=float)
    return (x - lo) / (hi - lo)


def load_transcript_lengths_from_fasta(path: Path) -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    with open_text(path) as fh:
        current_id = None
        seq_len = 0
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    lengths[current_id] = seq_len
                current_id = line[1:].split()[0].split(".", 1)[0]
                seq_len = 0
            else:
                seq_len += len(line)
        if current_id is not None:
            lengths[current_id] = seq_len
    return lengths


def load_transcript_sequences_from_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    with open_text(path) as fh:
        current_id = None
        chunks: List[str] = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(chunks).upper()
                current_id = line[1:].split()[0].split(".", 1)[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            seqs[current_id] = "".join(chunks).upper()
    return seqs


def looks_like_sample_col(col: str) -> bool:
    if col.startswith("Unnamed:"):
        return True
    return any(p.match(col) for p in SAMPLE_ID_PATTERNS)


def choose_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "transcript_id", "transcript_id_clean", "transcript_id_versioned", "gene_id", "gene_name",
        "gene_biotype", "gene_type", "transcript_type", "transcript_biotype", "strand", "transcript_length_nt",
        "median_tpm", "mean_tpm", "max_tpm", "prevalence_tpm_gt_0_1", "prevalence_tpm_gt_0_5",
        "prevalence_tpm_gt_1", "n_samples_gt_1",
    ]
    keep = [c for c in preferred if c in df.columns and not looks_like_sample_col(c)]
    if not keep:
        raise ValueError("No expected transcript summary columns found.")
    return df[keep].copy()


def load_transcript_summary(path: Path, verbose: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = choose_summary_columns(df)
    if "transcript_id" not in df.columns:
        if "transcript_id_clean" in df.columns:
            df["transcript_id"] = df["transcript_id_clean"]
        elif "transcript_id_versioned" in df.columns:
            df["transcript_id"] = df["transcript_id_versioned"].map(strip_version)
        else:
            raise ValueError("Transcript summary lacks usable transcript identifier column.")
    df["transcript_id"] = df["transcript_id"].map(strip_version)
    if "transcript_type" in df.columns and "transcript_biotype" not in df.columns:
        df = df.rename(columns={"transcript_type": "transcript_biotype"})
    if "gene_type" in df.columns and "gene_biotype" not in df.columns:
        df = df.rename(columns={"gene_type": "gene_biotype"})
    log(f"Transcript summary columns kept: {len(df.columns)}", verbose)
    return df.drop_duplicates(subset=["transcript_id"]).copy()


def load_orf_table(path: Path, verbose: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "transcript_id" not in df.columns:
        raise ValueError("ORF table must contain transcript_id.")
    df["transcript_id"] = df["transcript_id"].map(strip_version)
    log(f"Loaded ORFs: {len(df):,}", verbose)
    return df


def infer_coord_cols(df: pd.DataFrame) -> Tuple[str, str]:
    start_candidates = ["start_1based", "orf_start_nt", "start"]
    end_candidates = ["end_1based", "orf_end_nt", "end"]
    start_col = next((c for c in start_candidates if c in df.columns), None)
    end_col = next((c for c in end_candidates if c in df.columns), None)
    if start_col is None or end_col is None:
        raise ValueError("Could not find ORF coordinate columns.")
    return start_col, end_col


def infer_primary_nt_seq(row: pd.Series, tx_sequences: Dict[str, str]) -> Tuple[object, object]:
    tx_id = row["transcript_id"]
    tx_seq = tx_sequences.get(tx_id)
    s = row["orf_start_nt"]
    e = row["orf_end_nt"]
    if tx_seq is None or pd.isna(s) or pd.isna(e):
        return pd.NA, pd.NA
    s = int(s)
    e = int(e)
    if s < 1 or e > len(tx_seq) or s > e:
        return pd.NA, pd.NA
    reconstructed = tx_seq[s - 1:e]
    existing = row.get("nucleotide_sequence", pd.NA)
    existing_clean = clean_nt(existing) if pd.notna(existing) else ""
    if existing_clean:
        return reconstructed, existing_clean == reconstructed
    return reconstructed, pd.NA


def max_hydrophobic_run(peptide: str) -> int:
    peptide = clean_peptide(peptide)
    best = cur = 0
    for aa in peptide:
        if aa in HYDROPHOBIC_AA:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def estimate_charge(peptide: str, ph: float = 7.0) -> float:
    pep = clean_peptide(peptide)
    if not pep:
        return np.nan

    def frac_protonated_basic(pka: float) -> float:
        return 1.0 / (1.0 + 10 ** (ph - pka))

    def frac_deprotonated_acidic(pka: float) -> float:
        return 1.0 / (1.0 + 10 ** (pka - ph))

    charge = frac_protonated_basic(PKA_NTERM) - frac_deprotonated_acidic(PKA_CTERM)
    counts = Counter(pep)
    charge += counts["K"] * frac_protonated_basic(PKA_SIDE["K"])
    charge += counts["R"] * frac_protonated_basic(PKA_SIDE["R"])
    charge += counts["H"] * frac_protonated_basic(PKA_SIDE["H"])
    charge -= counts["D"] * frac_deprotonated_acidic(PKA_SIDE["D"])
    charge -= counts["E"] * frac_deprotonated_acidic(PKA_SIDE["E"])
    charge -= counts["C"] * frac_deprotonated_acidic(PKA_SIDE["C"])
    charge -= counts["Y"] * frac_deprotonated_acidic(PKA_SIDE["Y"])
    return float(charge)


def estimate_isoelectric_point(peptide: str) -> float:
    pep = clean_peptide(peptide)
    if not pep:
        return np.nan
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if estimate_charge(pep, mid) > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def compute_peptide_descriptors(peptide: str) -> Dict[str, float]:
    pep = clean_peptide(peptide)
    if not pep:
        return {
            "translated_peptide_length": np.nan,
            "translated_molecular_weight_da": np.nan,
            "translated_isoelectric_point": np.nan,
            "translated_gravy": np.nan,
            "translated_aromaticity": np.nan,
            "translated_net_charge_pH7": np.nan,
            "translated_fraction_hydrophobic": np.nan,
            "translated_fraction_charged": np.nan,
            "translated_max_hydrophobic_run": np.nan,
        }
    counts = Counter(pep)
    n = len(pep)
    return {
        "translated_peptide_length": n,
        "translated_molecular_weight_da": float(sum(AA_MW.get(aa, 0.0) for aa in pep) - 18.015 * max(n - 1, 0)),
        "translated_isoelectric_point": estimate_isoelectric_point(pep),
        "translated_gravy": float(np.mean([KD.get(aa, 0.0) for aa in pep])),
        "translated_aromaticity": safe_div(sum(counts[a] for a in AA_AROMATIC), n, np.nan),
        "translated_net_charge_pH7": estimate_charge(pep, 7.0),
        "translated_fraction_hydrophobic": safe_div(sum(counts[a] for a in HYDROPHOBIC_AA), n, np.nan),
        "translated_fraction_charged": safe_div(sum(counts[a] for a in CHARGED_AA), n, np.nan),
        "translated_max_hydrophobic_run": max_hydrophobic_run(pep),
    }


def compute_overlap_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["has_overlap_in_transcript"] = False
    out["nested_within_another_orf"] = False
    out["contains_another_orf"] = False
    out["max_overlap_fraction_with_any_orf"] = 0.0

    groups = out.groupby("transcript_id", sort=False).groups
    for _, idx in groups.items():
        locs = list(idx)
        if len(locs) < 2:
            continue
        starts = out.loc[locs, "orf_start_nt"].astype(int).to_numpy()
        ends = out.loc[locs, "orf_end_nt"].astype(int).to_numpy()
        lengths = out.loc[locs, "length_nt"].astype(float).to_numpy()
        group_overlap = np.zeros(len(locs), dtype=bool)
        group_nested = np.zeros(len(locs), dtype=bool)
        group_contains = np.zeros(len(locs), dtype=bool)
        group_best = np.zeros(len(locs), dtype=float)

        for i in range(len(locs)):
            s1, e1, l1 = starts[i], ends[i], lengths[i]
            for j in range(len(locs)):
                if i == j:
                    continue
                s2, e2, l2 = starts[j], ends[j], lengths[j]
                ov = max(0, min(e1, e2) - max(s1, s2) + 1)
                if ov > 0:
                    group_overlap[i] = True
                    group_best[i] = max(group_best[i], ov / max(l1, 1.0))
                if s2 <= s1 and e2 >= e1 and (s2 < s1 or e2 > e1):
                    group_nested[i] = True
                if s1 <= s2 and e1 >= e2 and (s1 < s2 or e1 > e2):
                    group_contains[i] = True

        out.loc[locs, "has_overlap_in_transcript"] = group_overlap
        out.loc[locs, "nested_within_another_orf"] = group_nested
        out.loc[locs, "contains_another_orf"] = group_contains
        out.loc[locs, "max_overlap_fraction_with_any_orf"] = group_best

    out["has_overlap_in_transcript"] = out["has_overlap_in_transcript"].astype("boolean")
    out["nested_within_another_orf"] = out["nested_within_another_orf"].astype("boolean")
    out["contains_another_orf"] = out["contains_another_orf"].astype("boolean")
    return out


def build_structural_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    length_norm = robust_minmax(out["length_nt"]).fillna(0.0)
    frac_norm = pd.to_numeric(out["orf_fraction_of_transcript"], errors="coerce").clip(lower=0, upper=1).fillna(0.0)
    codon_entropy_norm = robust_minmax(out["codon_usage_entropy"]).fillna(0.0)
    complexity_norm = pd.to_numeric(out["nucleotide_complexity"], errors="coerce").clip(0, 1).fillna(0.0)
    crowd_penalty = (1.0 / np.sqrt(pd.to_numeric(out["orfs_per_transcript"], errors="coerce").clip(lower=1))).fillna(0.0)
    overlap_penalty = 1.0 - pd.to_numeric(out["max_overlap_fraction_with_any_orf"], errors="coerce").clip(0, 1).fillna(0.0)
    first_triplet_bonus = out["first_triplet_class"].map({
        "canonical_start_like": 1.0, "near_cognate_like": 0.6, "other": 0.0, "stop": -0.5
    }).fillna(0.0)
    terminal_stop_bonus = out["terminal_triplet_is_stop"].fillna(False).astype(bool).astype(float)
    internal_stop_bonus = 1.0 - out["contains_internal_stop"].fillna(False).astype(bool).astype(float)
    longest_bonus = out["is_longest_orf_in_tx"].fillna(False).astype(bool).astype(float)

    out["structural_orf_score"] = (
        0.28 * length_norm +
        0.16 * frac_norm +
        0.12 * complexity_norm +
        0.10 * codon_entropy_norm +
        0.10 * crowd_penalty +
        0.08 * overlap_penalty +
        0.08 * longest_bonus +
        0.04 * first_triplet_bonus.clip(lower=0) +
        0.02 * terminal_stop_bonus +
        0.02 * internal_stop_bonus
    ).astype(float)
    out["structural_orf_rank_global"] = out["structural_orf_score"].rank(method="dense", ascending=False).astype("Int64")

    rep = out.sort_values(
        by=["transcript_id", "structural_orf_score", "length_nt", "orf_fraction_of_transcript", "max_overlap_fraction_with_any_orf"],
        ascending=[True, False, False, False, True],
    ).copy()
    rep["representative_orf_rank"] = rep.groupby("transcript_id").cumcount() + 1

    merge_keys = ["transcript_id"]
    if "raw_orf_id" in out.columns:
        merge_keys.append("raw_orf_id")
    elif "orf_index" in out.columns:
        merge_keys.append("orf_index")
    else:
        merge_keys.append("nucleotide_sequence")

    out = out.merge(rep[merge_keys + ["representative_orf_rank"]], on=merge_keys, how="left")
    out["is_representative_orf"] = out["representative_orf_rank"].eq(1).astype("boolean")
    out["structural_orf_rank_in_tx"] = out.groupby("transcript_id")["structural_orf_score"].rank(method="dense", ascending=False).astype("Int64")
    out["structural_confidence"] = out["structural_orf_score"].map(lambda x: "high" if x >= 0.70 else "medium" if x >= 0.50 else "low").astype("string")
    return out


def add_core_annotations(df: pd.DataFrame, tx_lengths: Dict[str, int], tx_sequences: Dict[str, str], min_nt_len: int, verbose: bool) -> pd.DataFrame:
    df = df.copy()
    start_col, end_col = infer_coord_cols(df)
    df["orf_start_nt"] = pd.to_numeric(df[start_col], errors="coerce")
    df["orf_end_nt"] = pd.to_numeric(df[end_col], errors="coerce")
    df["orf_span_nt"] = df["orf_end_nt"] - df["orf_start_nt"] + 1
    df["start_nt_0based"] = df["orf_start_nt"] - 1
    df["end_nt_0based_inclusive"] = df["orf_end_nt"] - 1

    if "transcript_length_nt" not in df.columns:
        df["transcript_length_nt"] = df["transcript_id"].map(tx_lengths)
    else:
        existing = pd.to_numeric(df["transcript_length_nt"], errors="coerce")
        df["transcript_length_nt"] = existing.fillna(df["transcript_id"].map(tx_lengths))

    reconstructed = df.apply(lambda row: infer_primary_nt_seq(row, tx_sequences), axis=1)
    df["reconstructed_nucleotide_sequence"] = reconstructed.map(lambda x: x[0])
    df["input_sequence_matches_reconstruction"] = reconstructed.map(lambda x: x[1]).astype("boolean")
    df["nucleotide_sequence_input"] = df.get("nucleotide_sequence", pd.Series(pd.NA, index=df.index))
    df["nucleotide_sequence"] = df["reconstructed_nucleotide_sequence"].fillna(df["nucleotide_sequence_input"])
    nt = df["nucleotide_sequence"].fillna("").astype(str).map(clean_nt)

    df["sequence_reconstructed_successfully"] = df["reconstructed_nucleotide_sequence"].notna().astype("boolean")
    df["orf_coordinates_valid"] = (
        df["orf_start_nt"].notna() & df["orf_end_nt"].notna() & (df["orf_start_nt"] >= 1) & (df["orf_end_nt"] >= df["orf_start_nt"])
    ).astype("boolean")
    df["orf_within_transcript_bounds"] = (
        df["orf_coordinates_valid"].fillna(False).astype(bool) & df["transcript_length_nt"].notna() & (df["orf_end_nt"] <= df["transcript_length_nt"])
    ).astype("boolean")

    df["length_nt"] = nt.str.len().where(nt.str.len() > 0, pd.to_numeric(df.get("orf_length_nt"), errors="coerce"))
    df["length_nt_matches_span"] = (pd.to_numeric(df["length_nt"], errors="coerce") == pd.to_numeric(df["orf_span_nt"], errors="coerce")).astype("boolean")
    df["passes_min_nt_filter"] = (pd.to_numeric(df["length_nt"], errors="coerce") >= min_nt_len).astype("boolean")
    df["length_mod_3"] = pd.to_numeric(df["length_nt"], errors="coerce") % 3
    df["length_mod3_ok"] = (df["length_mod_3"] == 0).astype("boolean")

    df["first_triplet"] = nt.str[:3].where(nt.str.len() >= 3, pd.NA)
    df["last_triplet"] = nt.str[-3:].where(nt.str.len() >= 3, pd.NA)
    df["first_triplet_class"] = df["first_triplet"].map(classify_first_triplet).astype("string")
    df["terminal_triplet_is_stop"] = df["last_triplet"].isin(list(STOP_CODONS)).astype("boolean")
    df["contains_internal_stop"] = nt.map(lambda x: count_internal_stops(x) > 0).astype("boolean")
    df["n_internal_stop_codons"] = nt.map(count_internal_stops).astype("Int64")

    df["gc_content"] = nt.map(gc_content)
    df["gc3_content"] = nt.map(gc3_content)
    df["at_content"] = 1.0 - pd.to_numeric(df["gc_content"], errors="coerce")
    df["nucleotide_complexity"] = nt.map(nucleotide_complexity)
    df["cpg_dinucleotide_fraction"] = nt.map(lambda x: dinucleotide_fraction(x, "CG"))
    df["longest_homopolymer_run_nt"] = nt.map(longest_homopolymer_run)
    df["codon_count"] = nt.map(lambda x: len(split_codons(x)))
    df["codon_usage_entropy"] = nt.map(codon_usage_entropy)

    df["distance_to_transcript_start_nt"] = df["orf_start_nt"] - 1
    df["distance_to_transcript_end_nt"] = df["transcript_length_nt"] - df["orf_end_nt"]
    df["orf_start_rel"] = df["orf_start_nt"] / df["transcript_length_nt"]
    df["orf_end_rel"] = df["orf_end_nt"] / df["transcript_length_nt"]
    df["orf_mid_rel"] = (df["orf_start_rel"] + df["orf_end_rel"]) / 2.0
    df["orf_fraction_of_transcript"] = pd.to_numeric(df["length_nt"], errors="coerce") / pd.to_numeric(df["transcript_length_nt"], errors="coerce")
    df["orf_region"] = pd.cut(df["orf_mid_rel"], bins=[-float("inf"), 1/3, 2/3, float("inf")], labels=["5prime", "middle", "3prime"]).astype("string")

    df["orfs_per_transcript"] = df.groupby("transcript_id")["transcript_id"].transform("size").astype("Int64")
    df["max_orf_length_nt_in_tx"] = df.groupby("transcript_id")["length_nt"].transform("max")
    df["orf_rank_by_length_in_tx"] = df.groupby("transcript_id")["length_nt"].rank(method="dense", ascending=False).astype("Int64")
    df["is_longest_orf_in_tx"] = df["orf_rank_by_length_in_tx"].eq(1).astype("boolean")
    df["length_nt_relative_to_longest_in_tx"] = pd.to_numeric(df["length_nt"], errors="coerce") / pd.to_numeric(df["max_orf_length_nt_in_tx"], errors="coerce")

    df = compute_overlap_features(df)

    translated = nt.map(lambda x: translate_nt(x, stop_at_first_stop=True))
    df["translated_peptide_sequence"] = translated
    peptide_desc = pd.DataFrame([compute_peptide_descriptors(x) for x in translated], index=df.index)
    df = pd.concat([df, peptide_desc], axis=1)

    df = build_structural_score(df)
    log(f"Annotated rows built: {len(df):,}", verbose)
    return df


def count_leaked_sample_cols(cols: Iterable[str]) -> int:
    return sum(looks_like_sample_col(str(c)) for c in cols)


def build_summary(df: pd.DataFrame) -> str:
    lines = [
        f"annotated_orfs\t{len(df)}",
        f"transcripts_represented\t{df['transcript_id'].nunique()}",
        f"columns\t{len(df.columns)}",
        f"likely_leaked_sample_columns\t{count_leaked_sample_cols(df.columns)}",
    ]
    for col in [
        "sequence_reconstructed_successfully", "input_sequence_matches_reconstruction", "orf_coordinates_valid",
        "orf_within_transcript_bounds", "length_nt_matches_span", "passes_min_nt_filter", "length_mod3_ok",
        "terminal_triplet_is_stop", "contains_internal_stop", "is_longest_orf_in_tx", "has_overlap_in_transcript",
        "nested_within_another_orf", "contains_another_orf", "is_representative_orf",
    ]:
        if col in df.columns:
            lines.append(f"{col}_frac\t{bool_mean_or_na(df[col])}")
    for col in [
        "length_nt", "gc_content", "gc3_content", "nucleotide_complexity", "cpg_dinucleotide_fraction",
        "codon_usage_entropy", "orf_fraction_of_transcript", "orfs_per_transcript", "max_overlap_fraction_with_any_orf",
        "structural_orf_score", "translated_peptide_length", "translated_gravy", "translated_net_charge_pH7",
        "translated_isoelectric_point",
    ]:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            lines.append(f"{col}_median\t{x.median()}")
            lines.append(f"{col}_mean\t{x.mean()}")
    if "structural_confidence" in df.columns:
        counts = df["structural_confidence"].value_counts(dropna=False).to_dict()
        for key in ["high", "medium", "low"]:
            lines.append(f"structural_confidence_{key}\t{counts.get(key, 0)}")
    return "\n".join(map(str, lines)) + "\n"


def save_outputs(df: pd.DataFrame, output_tsv: Path, output_parquet: Optional[Path], summary_out: Path, verbose: bool) -> None:
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_tsv, sep="\t", index=False)
    log(f"Saved TSV: {output_tsv}", verbose)
    if output_parquet:
        try:
            df.to_parquet(output_parquet, index=False)
            log(f"Saved parquet: {output_parquet}", verbose)
        except Exception as e:
            warn("Parquet export skipped: " + str(e))
    summary_out.write_text(build_summary(df), encoding="utf-8")
    log(f"Saved summary: {summary_out}", verbose)


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_yaml_config(args.config)
    annotate_dir = resolve_path(get_nested(cfg, "stage_outputs", "annotate_dir"), cfg)
    if args.orf_table is None:
        getorf_dir = get_nested(cfg, "stage_outputs", "getorf_dir")
        if getorf_dir:
            args.orf_table = resolve_path(Path(getorf_dir) / "getorf_stop_stop_nt_candidates.tsv", cfg)
    if args.transcript_summary is None:
        prepare_dir = get_nested(cfg, "stage_outputs", "prepare_dir")
        if prepare_dir:
            args.transcript_summary = resolve_path(Path(prepare_dir) / "muscle_selected_lncRNA_transcript_summary.tsv", cfg)
    if args.transcripts_fasta is None:
        prepare_dir = get_nested(cfg, "stage_outputs", "prepare_dir")
        if prepare_dir:
            args.transcripts_fasta = resolve_path(Path(prepare_dir) / "muscle_selected_lncRNA_transcripts_versioned.fa", cfg)
    if args.output_tsv is None and annotate_dir is not None:
        args.output_tsv = annotate_dir / "07_annotated_orf_candidates.tsv"
    if args.output_parquet is None and annotate_dir is not None:
        args.output_parquet = annotate_dir / "07_annotated_orf_candidates.parquet"
    if args.summary_out is None and annotate_dir is not None:
        args.summary_out = annotate_dir / "07_annotation_summary.txt"
    if args.min_nt_len is None:
        args.min_nt_len = int(get_nested(cfg, "annotate", "min_nt_len", default=60))
    required = ["orf_table", "transcript_summary", "transcripts_fasta", "output_tsv", "summary_out"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"Missing required arguments after config resolution: {missing}")
    return args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Annotate stop-stop nucleotide ORF candidates with structural and transcript-context features.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--orf-table", default=None)
    p.add_argument("--transcript-summary", default=None)
    p.add_argument("--transcripts-fasta", default=None)
    p.add_argument("--output-tsv", default=None)
    p.add_argument("--output-parquet", default=None)
    p.add_argument("--summary-out", default=None)
    p.add_argument("--min-nt-len", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = apply_config_defaults(parse_args())
    orf_df = load_orf_table(Path(args.orf_table), args.verbose)
    tx_summary = load_transcript_summary(Path(args.transcript_summary), args.verbose)
    tx_lengths = load_transcript_lengths_from_fasta(Path(args.transcripts_fasta))
    tx_sequences = load_transcript_sequences_from_fasta(Path(args.transcripts_fasta))

    merged = orf_df.copy()
    merged["transcript_id"] = merged["transcript_id"].map(strip_version)
    merged = merged.merge(tx_summary, on="transcript_id", how="left", validate="many_to_one")

    annotated = add_core_annotations(
        merged,
        tx_lengths=tx_lengths,
        tx_sequences=tx_sequences,
        min_nt_len=args.min_nt_len,
        verbose=args.verbose,
    )

    save_outputs(
        annotated,
        Path(args.output_tsv),
        Path(args.output_parquet) if args.output_parquet else None,
        Path(args.summary_out),
        args.verbose,
    )
    log("Done.", args.verbose)


if __name__ == "__main__":
    main()
