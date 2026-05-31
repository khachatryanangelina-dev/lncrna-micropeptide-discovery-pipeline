#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

from lncrna_micropeptides.pipeline_config import get_nested, load_yaml_config, resolve_path

STOP_CODONS = {"TAA", "TAG", "TGA"}
DEFAULT_FIND_MODE = 2  # Nucleic sequences between STOP codons


def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def read_text_auto(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "rt")


def fasta_iter(path: Path) -> Iterator[Tuple[str, str]]:
    header: Optional[str] = None
    chunks: List[str] = []
    with read_text_auto(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)


def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def parse_input_header(description: str) -> Dict[str, Optional[str]]:
    parts = description.split("|")
    return {
        "transcript_id": parts[0] if len(parts) > 0 else None,
        "gene_id": parts[1] if len(parts) > 1 else None,
        "gene_name": parts[5] if len(parts) > 5 else None,
        "transcript_length_header": parts[6] if len(parts) > 6 else None,
        "transcript_type": parts[7] if len(parts) > 7 else None,
    }


def prepare_getorf_input(input_fasta: Path, output_fasta: Path, output_meta_tsv: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with open(output_fasta, "w") as out:
        for header, seq in fasta_iter(input_fasta):
            meta = parse_input_header(header)
            transcript_id = meta["transcript_id"]
            if not transcript_id:
                warn(f"Skipping record with unparsable transcript_id header: {header[:120]}")
                continue

            seq = seq.upper()
            out.write(f">{transcript_id}\n{wrap_fasta(seq)}\n")
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "gene_id": meta["gene_id"],
                    "gene_name": meta["gene_name"],
                    "transcript_type": meta["transcript_type"],
                    "transcript_length_header": meta["transcript_length_header"],
                    "transcript_length_nt": len(seq),
                    "input_header": header,
                }
            )

    meta_df = pd.DataFrame(rows)
    meta_df.to_csv(output_meta_tsv, sep="\t", index=False)
    return meta_df


def check_getorf_available(explicit_path: Optional[str] = None) -> str:
    exe = explicit_path or "getorf"
    resolved = shutil.which(exe) if explicit_path is None else explicit_path
    if explicit_path is None and resolved is None:
        raise FileNotFoundError(
            "GETORF executable was not found in PATH. Install EMBOSS or pass --getorf-bin."
        )
    return resolved or exe


def run_getorf(
    getorf_bin: str,
    input_fasta: Path,
    output_fasta: Path,
    min_nt: int,
    max_nt: int,
    find_mode: int = DEFAULT_FIND_MODE,
    reverse: str = "N",
    genetic_code: int = 0,
) -> None:
    cmd = [
        getorf_bin,
        "-sequence",
        str(input_fasta),
        "-outseq",
        str(output_fasta),
        "-table",
        str(genetic_code),
        "-find",
        str(find_mode),
        "-minsize",
        str(min_nt),
        "-maxsize",
        str(max_nt),
        "-reverse",
        str(reverse),
    ]
    log("Running GETORF:")
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)


GETORF_HEADER_PATTERNS = [
    re.compile(r"^(.+?)_(\d+)\s+\[(\d+)\s*-\s*(\d+)\]$"),
    re.compile(r"^(.+?)_(\d+)\s*\[(\d+)\s*-\s*(\d+)\]$"),
]


def parse_getorf_header(header: str) -> Dict[str, Optional[object]]:
    cleaned = " ".join(header.split())
    for pattern in GETORF_HEADER_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            transcript_id = match.group(1)
            orf_index = int(match.group(2))
            start_1based = int(match.group(3))
            end_1based = int(match.group(4))
            return {
                "raw_orf_id": f"{transcript_id}_{orf_index}",
                "transcript_id": transcript_id,
                "orf_index": orf_index,
                "start_1based": start_1based,
                "end_1based": end_1based,
            }

    warn(f"Could not parse GETORF header: {header}")
    return {
        "raw_orf_id": None,
        "transcript_id": None,
        "orf_index": None,
        "start_1based": None,
        "end_1based": None,
    }


def infer_frame(start_1based: Optional[int]) -> Optional[int]:
    if start_1based is None:
        return None
    return (start_1based - 1) % 3


def classify_first_triplet(nt_seq: str) -> Optional[str]:
    if len(nt_seq) < 3:
        return None
    triplet = nt_seq[:3]
    if triplet == "ATG":
        return "canonical_start"
    if triplet in {"CTG", "GTG", "TTG", "ATA", "ATC", "ATT", "ACG"}:
        return "near_cognate_start"
    if triplet in STOP_CODONS:
        return "stop_codon"
    return "other"


def gc_content(seq: str) -> Optional[float]:
    if not seq:
        return None
    seq = seq.upper()
    valid = sum(base in {"A", "C", "G", "T"} for base in seq)
    if valid == 0:
        return None
    gc = sum(base in {"G", "C"} for base in seq)
    return gc / valid


def parse_getorf_output(getorf_output_fasta: Path, meta_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for header, seq in fasta_iter(getorf_output_fasta):
        parsed = parse_getorf_header(header)
        nt_seq = seq.strip().upper()
        start_1based = parsed["start_1based"]
        end_1based = parsed["end_1based"]
        first_triplet = nt_seq[:3] if len(nt_seq) >= 3 else None
        last_triplet = nt_seq[-3:] if len(nt_seq) >= 3 else None
        length_nt = len(nt_seq)

        rows.append(
            {
                "method": "getorf_stop_stop_nt",
                "getorf_header": header,
                "raw_orf_id": parsed["raw_orf_id"],
                "transcript_id": parsed["transcript_id"],
                "orf_index": parsed["orf_index"],
                "start_1based": start_1based,
                "end_1based": end_1based,
                "start_nt_0based": (start_1based - 1) if start_1based is not None else pd.NA,
                "end_nt_0based_inclusive": (end_1based - 1) if end_1based is not None else pd.NA,
                "frame": infer_frame(start_1based),
                "orf_length_nt": length_nt,
                "length_mod_3": length_nt % 3 if length_nt else pd.NA,
                "first_triplet": first_triplet,
                "last_triplet": last_triplet,
                "first_triplet_class": classify_first_triplet(nt_seq),
                "terminal_triplet_is_stop": last_triplet in STOP_CODONS if last_triplet else False,
                "contains_internal_stop": any(
                    nt_seq[i : i + 3] in STOP_CODONS for i in range(0, max(length_nt - 3, 0), 3)
                ),
                "gc_content": gc_content(nt_seq),
                "nucleotide_sequence": nt_seq,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    merged = df.merge(meta_df, on="transcript_id", how="left")
    if "transcript_length_nt" in merged.columns:
        merged["orf_fraction_of_transcript"] = merged["orf_length_nt"] / merged["transcript_length_nt"]
        merged["relative_start"] = merged["start_1based"] / merged["transcript_length_nt"]
        merged["relative_end"] = merged["end_1based"] / merged["transcript_length_nt"]
    else:
        merged["orf_fraction_of_transcript"] = pd.NA
        merged["relative_start"] = pd.NA
        merged["relative_end"] = pd.NA

    return merged


def add_transcript_level_orf_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["n_orfs_in_transcript"] = out.groupby("transcript_id")["raw_orf_id"].transform("count")
    out["max_orf_length_nt_in_transcript"] = out.groupby("transcript_id")["orf_length_nt"].transform("max")
    out["is_longest_orf_in_transcript"] = (
        out["orf_length_nt"] == out["max_orf_length_nt_in_transcript"]
    )
    out["length_rank_within_transcript"] = out.groupby("transcript_id")["orf_length_nt"].rank(
        method="dense", ascending=False
    )
    return out


def save_run_summary(
    out_path: Path,
    input_fasta: Path,
    min_nt: int,
    max_nt: int,
    find_mode: int,
    reverse: str,
    genetic_code: int,
    parsed_df: pd.DataFrame,
) -> None:
    lines = [
        "GETORF run summary",
        f"input_fasta\t{input_fasta}",
        f"find_mode\t{find_mode}",
        f"genetic_code\t{genetic_code}",
        f"reverse\t{reverse}",
        f"min_nt\t{min_nt}",
        f"max_nt\t{max_nt}",
        f"n_orfs\t{len(parsed_df)}",
        f"n_transcripts_with_orfs\t{parsed_df['transcript_id'].nunique() if not parsed_df.empty else 0}",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def apply_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    cfg = load_yaml_config(args.config)

    if args.input_fasta is None:
        prepare_dir = get_nested(cfg, "stage_outputs", "prepare_dir")
        if prepare_dir:
            args.input_fasta = resolve_path(
                Path(prepare_dir) / "muscle_selected_lncRNA_transcripts_versioned.fa", cfg
            )

    if args.output_dir is None:
        args.output_dir = resolve_path(get_nested(cfg, "stage_outputs", "getorf_dir"), cfg)

    if args.min_nt is None:
        args.min_nt = int(get_nested(cfg, "getorf", "min_nt", default=60))

    if args.max_nt is None:
        args.max_nt = int(get_nested(cfg, "getorf", "max_nt", default=450))

    if args.find_mode is None:
        args.find_mode = int(get_nested(cfg, "getorf", "find_mode", default=DEFAULT_FIND_MODE))

    if args.reverse is None:
        args.reverse = str(get_nested(cfg, "getorf", "reverse", default="N"))

    if args.genetic_code is None:
        args.genetic_code = int(get_nested(cfg, "getorf", "genetic_code", default=0))

    if args.input_fasta is None or args.output_dir is None:
        raise ValueError("input-fasta and output-dir must be provided either directly or through config")

    return args


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run EMBOSS getorf on lncRNA transcript FASTA in nucleotide stop-to-stop mode."
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config file with pipeline paths and parameters.")
    p.add_argument("--input-fasta", default=None, help="Final transcript FASTA from preparation step.")
    p.add_argument("--output-dir", default=None, help="Directory for GETORF outputs.")
    p.add_argument("--getorf-bin", default=None, help="Path to getorf executable.")
    p.add_argument("--min-nt", type=int, default=None, help="Minimum ORF length in nucleotides.")
    p.add_argument("--max-nt", type=int, default=None, help="Maximum ORF length in nucleotides.")
    p.add_argument(
        "--find-mode",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4, 5, 6],
        help="GETORF -find mode. Default: 2 (nucleic sequences between STOP codons).",
    )
    p.add_argument("--reverse", default=None, choices=["Y", "N"], help="Search reverse complement too.")
    p.add_argument("--genetic-code", type=int, default=None, help="EMBOSS genetic code table. Default: 0.")
    p.add_argument("--skip-run", action="store_true", help="Skip GETORF execution and parse existing output.")
    p.add_argument(
        "--existing-getorf-output",
        default=None,
        help="Existing GETORF nucleotide FASTA output to parse when --skip-run is used.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    args = apply_config_defaults(args)

    input_fasta = Path(args.input_fasta)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.min_nt <= 0 or args.max_nt <= 0:
        raise ValueError("min-nt and max-nt must be positive integers")
    if args.min_nt > args.max_nt:
        raise ValueError("min-nt must be less than or equal to max-nt")

    getorf_input_fasta = output_dir / "muscle_selected_lncRNA_transcripts_getorf_input.fa"
    getorf_input_meta = output_dir / "getorf_input_transcript_metadata.tsv"
    getorf_output_fasta = output_dir / "getorf_output_nt_stop_stop.fa"
    parsed_output_tsv = output_dir / "getorf_stop_stop_nt_candidates.tsv"
    summary_txt = output_dir / "getorf_run_summary.txt"

    meta_df = prepare_getorf_input(
        input_fasta=input_fasta,
        output_fasta=getorf_input_fasta,
        output_meta_tsv=getorf_input_meta,
    )

    if args.skip_run:
        if not args.existing_getorf_output:
            raise ValueError("--skip-run requires --existing-getorf-output")
        getorf_output_fasta = Path(args.existing_getorf_output)
    else:
        getorf_bin = check_getorf_available(args.getorf_bin)
        run_getorf(
            getorf_bin=getorf_bin,
            input_fasta=getorf_input_fasta,
            output_fasta=getorf_output_fasta,
            min_nt=args.min_nt,
            max_nt=args.max_nt,
            find_mode=args.find_mode,
            reverse=args.reverse,
            genetic_code=args.genetic_code,
        )

    parsed_df = parse_getorf_output(getorf_output_fasta, meta_df)
    parsed_df = add_transcript_level_orf_stats(parsed_df)
    parsed_df.to_csv(parsed_output_tsv, sep="\t", index=False)

    save_run_summary(
        out_path=summary_txt,
        input_fasta=input_fasta,
        min_nt=args.min_nt,
        max_nt=args.max_nt,
        find_mode=args.find_mode,
        reverse=args.reverse,
        genetic_code=args.genetic_code,
        parsed_df=parsed_df,
    )

    log(f"Saved nucleotide GETORF output: {getorf_output_fasta}")
    log(f"Saved parsed candidate table: {parsed_output_tsv}")
    log(f"Saved run summary: {summary_txt}")
    log("Done.")


if __name__ == "__main__":
    main()
