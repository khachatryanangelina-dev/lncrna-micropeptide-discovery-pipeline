from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_TOP_LEVEL_KEYS = [
    "paths",
    "stage_outputs",
]


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("Config path must be provided.")

    cfg_path = Path(path).expanduser().resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top level: {cfg_path}")

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            raise ValueError(f"Missing required config section: {key}")

    data["_config_path"] = cfg_path
    data["_config_dir"] = cfg_path.parent

    return data


def get_nested(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path(
    value: str | Path | None,
    cfg: dict[str, Any] | None = None,
    must_exist: bool = False,
) -> Path | None:
    if value is None:
        return None

    p = Path(value).expanduser()

    if not p.is_absolute():
        cfg_dir = cfg.get("_config_dir") if cfg else None
        if cfg_dir:
            p = Path(cfg_dir) / p

    p = p.resolve()

    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    return p


def validate_paths(cfg: dict[str, Any]) -> None:
    """
    Validate key input files and output directories declared in config.

    This function intentionally checks only the stable pipeline-level paths.
    Stage-specific scripts may still validate their own optional inputs.
    """

    required_input_paths = [
        ("paths", "gencode_gtf"),
        ("paths", "gencode_transcripts_fasta"),
        ("paths", "gtex_gene_tpm"),
        ("paths", "gtex_transcript_tpm"),
    ]

    missing_keys: list[str] = []

    for keys in required_input_paths:
        value = get_nested(cfg, *keys)
        dotted = ".".join(keys)

        if value is None:
            missing_keys.append(dotted)
            continue

        resolve_path(value, cfg, must_exist=True)

    if missing_keys:
        raise ValueError(
            "Missing required config path(s): " + ", ".join(missing_keys)
        )

    output_sections = [
        ("stage_outputs", "prepare_dir"),
        ("stage_outputs", "getorf_dir"),
        ("stage_outputs", "annotate_dir"),
        ("stage_outputs", "score_dir"),
    ]

    for keys in output_sections:
        value = get_nested(cfg, *keys)
        if value is not None:
            path = resolve_path(value, cfg, must_exist=False)
            path.mkdir(parents=True, exist_ok=True)