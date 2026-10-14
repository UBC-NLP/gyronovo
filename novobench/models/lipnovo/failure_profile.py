"""Validation failure profiling utilities for the retained workflows."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

from .sequence import normalize_lipnovo_sequence


TOKEN_PATTERN = re.compile(r"[A-Z](?:\([^)]*\))?")
LENGTH_BUCKETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("len_le_10", None, 10),
    ("len_11_15", 11, 15),
    ("len_16_20", 16, 20),
    ("len_gt_20", 21, None),
)


@dataclass(frozen=True)
class PredictionExample:
    target: str
    prediction: str
    score: float
    pep_match: bool | None = None


def tokenize_peptide(sequence: str | None) -> list[str]:
    text = "" if sequence is None else str(sequence).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    text = str(normalize_lipnovo_sequence(text))
    tokens = TOKEN_PATTERN.findall(text)
    if "".join(tokens) == text:
        return tokens
    return list(text)


def is_ptm_token(token: str) -> bool:
    return "(" in token or ")" in token or token.startswith(("+", "-"))


def normalize_tokens(
    tokens: Sequence[str],
    *,
    strip_ptm: bool = False,
    collapse_il: bool = False,
) -> list[str]:
    normalized: list[str] = []
    for token in tokens:
        current = token
        if strip_ptm:
            current = current[0] if current else current
        if collapse_il and current in {"I", "L"}:
            current = "L"
        normalized.append(current)
    return normalized


def levenshtein_distance(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) == 0:
        return len(right)
    if len(right) == 0:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            substitution = previous[j - 1] + int(left_token != right_token)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def is_single_adjacent_swap(left: Sequence[str], right: Sequence[str]) -> bool:
    if len(left) != len(right) or len(left) < 2:
        return False
    mismatched = [idx for idx, (l_token, r_token) in enumerate(zip(left, right)) if l_token != r_token]
    if len(mismatched) != 2 or mismatched[1] != mismatched[0] + 1:
        return False
    i, j = mismatched
    return left[i] == right[j] and left[j] == right[i]


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _score_to_float(value: Any) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _bucket_name(length: int) -> str:
    for name, minimum, maximum in LENGTH_BUCKETS:
        if minimum is not None and length < minimum:
            continue
        if maximum is not None and length > maximum:
            continue
        return name
    return "len_unknown"


def _make_example(record: Mapping[str, Any]) -> PredictionExample:
    pep_match_raw = record.get("pep_matches", None)
    if isinstance(pep_match_raw, str):
        pep_match = pep_match_raw.strip().lower() == "true"
    elif pep_match_raw is None:
        pep_match = None
    else:
        pep_match = bool(pep_match_raw)
    return PredictionExample(
        target=str(record.get("peptides_true", "")),
        prediction=str(record.get("peptides_pred", "")),
        score=_score_to_float(record.get("peptides_score", float("-inf"))),
        pep_match=pep_match,
    )


def build_failure_profile(
    records: Iterable[Mapping[str, Any] | PredictionExample],
    *,
    high_conf_threshold: float = 0.98,
) -> dict[str, Any]:
    examples: list[PredictionExample] = []
    for record in records:
        if isinstance(record, PredictionExample):
            examples.append(record)
        else:
            examples.append(_make_example(record))

    total = len(examples)
    if total == 0:
        return {
            "n_predictions": 0,
            "peptide_match_rate": 0.0,
            "length_buckets": {},
            "ptm_vs_non_ptm": {},
            "wrong_prediction_analysis": {},
            "curriculum_priorities": {},
        }

    n_matches = 0
    length_stats: dict[str, dict[str, float]] = {
        name: {"count": 0.0, "match_count": 0.0, "match_rate": 0.0} for name, _, _ in LENGTH_BUCKETS
    }
    ptm_stats = {
        "ptm": {"count": 0.0, "match_count": 0.0, "match_rate": 0.0},
        "non_ptm": {"count": 0.0, "match_count": 0.0, "match_rate": 0.0},
    }

    wrong_count = 0
    near_miss_wrong = 0
    adjacent_swap_wrong = 0
    first_correct_wrong = 0
    last_correct_wrong = 0
    high_conf_wrong = 0

    for example in examples:
        target_tokens = tokenize_peptide(example.target)
        prediction_tokens = tokenize_peptide(example.prediction)
        pep_match = (
            example.pep_match
            if example.pep_match is not None
            else target_tokens == prediction_tokens
        )
        n_matches += int(pep_match)

        target_len = len(target_tokens)
        bucket = _bucket_name(target_len)
        length_stats[bucket]["count"] += 1
        length_stats[bucket]["match_count"] += int(pep_match)

        has_ptm = any(is_ptm_token(token) for token in target_tokens)
        ptm_key = "ptm" if has_ptm else "non_ptm"
        ptm_stats[ptm_key]["count"] += 1
        ptm_stats[ptm_key]["match_count"] += int(pep_match)

        if pep_match:
            continue

        wrong_count += 1
        target_norm = normalize_tokens(target_tokens, strip_ptm=True, collapse_il=True)
        prediction_norm = normalize_tokens(prediction_tokens, strip_ptm=True, collapse_il=True)

        if levenshtein_distance(target_norm, prediction_norm) <= 2:
            near_miss_wrong += 1
        if is_single_adjacent_swap(target_norm, prediction_norm):
            adjacent_swap_wrong += 1
        if target_norm and prediction_norm and target_norm[0] == prediction_norm[0]:
            first_correct_wrong += 1
        if target_norm and prediction_norm and target_norm[-1] == prediction_norm[-1]:
            last_correct_wrong += 1
        if example.score >= high_conf_threshold:
            high_conf_wrong += 1

    for bucket_values in length_stats.values():
        bucket_values["match_rate"] = _safe_rate(bucket_values["match_count"], bucket_values["count"])
    for bucket_values in ptm_stats.values():
        bucket_values["match_rate"] = _safe_rate(bucket_values["match_count"], bucket_values["count"])

    peptide_match_rate = _safe_rate(n_matches, total)
    wrong_rate = _safe_rate(wrong_count, total)
    near_miss_wrong_rate = _safe_rate(near_miss_wrong, wrong_count)
    adjacent_swap_wrong_rate = _safe_rate(adjacent_swap_wrong, wrong_count)
    first_correct_wrong_rate = _safe_rate(first_correct_wrong, wrong_count)
    last_correct_wrong_rate = _safe_rate(last_correct_wrong, wrong_count)
    high_conf_wrong_rate_all = _safe_rate(high_conf_wrong, total)
    high_conf_wrong_rate_wrong = _safe_rate(high_conf_wrong, wrong_count)

    len_gt_20_rate = length_stats["len_gt_20"]["match_rate"]
    ptm_gap = max(ptm_stats["non_ptm"]["match_rate"] - ptm_stats["ptm"]["match_rate"], 0.0)
    prefix_gap = max(last_correct_wrong_rate - first_correct_wrong_rate, 0.0)
    curriculum_priorities = {
        "long_peptides": min(max(1.0 - len_gt_20_rate, 0.0), 1.0),
        "ptm_peptides": min(max(max(1.0 - ptm_stats["ptm"]["match_rate"], ptm_gap), 0.0), 1.0),
        "near_miss_predictions": min(max(near_miss_wrong_rate, 0.0), 1.0),
        "adjacent_swap_predictions": min(max(adjacent_swap_wrong_rate, 0.0), 1.0),
        "prefix_weak_failures": min(max(prefix_gap, 0.0), 1.0),
        "high_confidence_wrong": min(max(high_conf_wrong_rate_wrong, 0.0), 1.0),
    }

    return {
        "n_predictions": total,
        "peptide_match_rate": peptide_match_rate,
        "length_buckets": length_stats,
        "ptm_vs_non_ptm": ptm_stats,
        "wrong_prediction_analysis": {
            "wrong_count": wrong_count,
            "wrong_rate": wrong_rate,
            "near_miss_edit_distance_le_2_count": near_miss_wrong,
            "near_miss_edit_distance_le_2_rate": near_miss_wrong_rate,
            "adjacent_swap_count": adjacent_swap_wrong,
            "adjacent_swap_rate": adjacent_swap_wrong_rate,
            "first_residue_correct_count": first_correct_wrong,
            "first_residue_correct_rate": first_correct_wrong_rate,
            "last_residue_correct_count": last_correct_wrong,
            "last_residue_correct_rate": last_correct_wrong_rate,
            "high_confidence_wrong_count": high_conf_wrong,
            "high_confidence_wrong_rate_all": high_conf_wrong_rate_all,
            "high_confidence_wrong_rate_wrong": high_conf_wrong_rate_wrong,
            "high_confidence_threshold": float(high_conf_threshold),
            "normalization": {
                "strip_ptm_annotations": True,
                "collapse_i_l": True,
            },
        },
        "curriculum_priorities": curriculum_priorities,
    }


def flatten_failure_profile_for_logging(profile: Mapping[str, Any]) -> dict[str, float]:
    wrong = profile.get("wrong_prediction_analysis", {})
    length_buckets = profile.get("length_buckets", {})
    ptm_stats = profile.get("ptm_vs_non_ptm", {})
    priorities = profile.get("curriculum_priorities", {})
    return {
        "peptide_match_rate": float(profile.get("peptide_match_rate", 0.0)),
        "len_le_10_match_rate": float(length_buckets.get("len_le_10", {}).get("match_rate", 0.0)),
        "len_11_15_match_rate": float(length_buckets.get("len_11_15", {}).get("match_rate", 0.0)),
        "len_16_20_match_rate": float(length_buckets.get("len_16_20", {}).get("match_rate", 0.0)),
        "len_gt_20_match_rate": float(length_buckets.get("len_gt_20", {}).get("match_rate", 0.0)),
        "ptm_match_rate": float(ptm_stats.get("ptm", {}).get("match_rate", 0.0)),
        "non_ptm_match_rate": float(ptm_stats.get("non_ptm", {}).get("match_rate", 0.0)),
        "near_miss_wrong_rate": float(wrong.get("near_miss_edit_distance_le_2_rate", 0.0)),
        "adjacent_swap_wrong_rate": float(wrong.get("adjacent_swap_rate", 0.0)),
        "first_residue_correct_wrong_rate": float(wrong.get("first_residue_correct_rate", 0.0)),
        "last_residue_correct_wrong_rate": float(wrong.get("last_residue_correct_rate", 0.0)),
        "high_confidence_wrong_rate": float(wrong.get("high_confidence_wrong_rate_all", 0.0)),
        "priority_long_peptides": float(priorities.get("long_peptides", 0.0)),
        "priority_ptm_peptides": float(priorities.get("ptm_peptides", 0.0)),
        "priority_near_miss": float(priorities.get("near_miss_predictions", 0.0)),
        "priority_adjacent_swap": float(priorities.get("adjacent_swap_predictions", 0.0)),
        "priority_prefix_weak": float(priorities.get("prefix_weak_failures", 0.0)),
        "priority_high_conf_wrong": float(priorities.get("high_confidence_wrong", 0.0)),
    }
