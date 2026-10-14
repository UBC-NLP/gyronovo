"""Peptide sequence normalization helpers for LIPNovo."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import re


MODIFICATION_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("M(+15.99)", "M(ox)", "M(Oxidation)", "M(oxidation)"),
    ("N(+.98)", "N(Deamidation)", "N(deamidation)", "N(+0.98)"),
    ("Q(+.98)", "Q(Deamidation)", "Q(deamidation)", "Q(+0.98)"),
    ("C(+57.02)", "C(Carbamidomethyl)", "C(carbamidomethyl)", "C(+57.021)"),
)
TOKEN_PATTERN = re.compile(r"[A-Z](?:\([^)]*\))?")


def _residue_keys(residues: Mapping[str, float] | Iterable[str] | None) -> set[str] | None:
    if residues is None:
        return None
    if isinstance(residues, Mapping):
        return {str(key) for key in residues}
    return {str(key) for key in residues}


def normalize_lipnovo_sequence(
    sequence: str | Sequence[str],
    residues: Mapping[str, float] | Iterable[str] | None = None,
) -> str | list[str]:
    """Normalize common equivalent PTM spellings to the active residue vocabulary."""
    residue_keys = _residue_keys(residues)

    def canonical_token(group: tuple[str, ...]) -> str | None:
        if residue_keys is None:
            return group[0]
        for token in group:
            if token in residue_keys:
                return token
        return None

    aliases: dict[str, str] = {}
    for group in MODIFICATION_ALIAS_GROUPS:
        canonical = canonical_token(group)
        if canonical is None:
            continue
        for token in group:
            if token != canonical:
                aliases[token] = canonical
    if residue_keys is not None and "C" not in residue_keys and "C(+57.02)" in residue_keys:
        aliases["C"] = "C(+57.02)"

    def normalize_token(token: str) -> str:
        return aliases.get(str(token), str(token))

    if isinstance(sequence, str):
        text = str(sequence)
        tokens = TOKEN_PATTERN.findall(text)
        if tokens and "".join(tokens) == text:
            return "".join(normalize_token(token) for token in tokens)
        for alias in sorted((key for key in aliases if len(key) > 1), key=len, reverse=True):
            text = text.replace(alias, aliases[alias])
        return text

    return [normalize_token(str(token)) for token in sequence]
