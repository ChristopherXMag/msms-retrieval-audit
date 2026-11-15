from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable, Literal, MutableMapping

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

SmilesMode = Literal["raw", "v15_raw", "canonical_isomeric", "canonical_nonisomeric"]


@dataclass(frozen=True)
class SmilesProfile:
    raw_smiles: str | None
    valid: bool
    canonical_isomeric: str | None = None
    canonical_nonisomeric: str | None = None
    formula: str | None = None
    inchi_key: str | None = None
    inchi_key_first14: str | None = None
    formal_charge: int | None = None
    fragment_count: int | None = None
    error: str | None = None

    @property
    def multifragment(self) -> bool:
        return bool(self.fragment_count is not None and self.fragment_count > 1)


def _clean_smiles(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    value = str(smiles).strip()
    return value or None


@lru_cache(maxsize=524288)
def profile_smiles(smiles: str | None) -> SmilesProfile:
    value = _clean_smiles(smiles)
    if value is None:
        return SmilesProfile(raw_smiles=smiles, valid=False, error="empty_smiles")

    mol = Chem.MolFromSmiles(value)
    if mol is None:
        return SmilesProfile(raw_smiles=value, valid=False, error="rdkit_parse_failed")

    inchi_key = None
    first14 = None
    try:
        inchi_key = Chem.MolToInchiKey(mol)
        first14 = inchi_key[:14] if inchi_key else None
    except Exception:
        inchi_key = None
        first14 = None

    return SmilesProfile(
        raw_smiles=value,
        valid=True,
        canonical_isomeric=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        canonical_nonisomeric=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        inchi_key=inchi_key,
        inchi_key_first14=first14,
        formal_charge=sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        fragment_count=len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)),
    )


def first14(inchi_key: str | None) -> str | None:
    value = _clean_smiles(inchi_key)
    return value[:14] if value else None


def canonical_smiles(smiles: str | None, *, isomeric: bool = True) -> str | None:
    profile = profile_smiles(smiles)
    if not profile.valid:
        return None
    return profile.canonical_isomeric if isomeric else profile.canonical_nonisomeric


def normalize_smiles(
    smiles: str | None,
    mode: SmilesMode = "canonical_isomeric",
    *,
    drop_invalid: bool = True,
    drop_multifragment: bool = False,
) -> str | None:
    profile = profile_smiles(smiles)
    if not profile.valid:
        return None if drop_invalid else _clean_smiles(smiles)
    if drop_multifragment and profile.multifragment:
        return None
    if mode in {"raw", "v15_raw"}:
        return profile.raw_smiles
    if mode == "canonical_isomeric":
        return profile.canonical_isomeric
    if mode == "canonical_nonisomeric":
        return profile.canonical_nonisomeric
    raise ValueError(f"Unsupported SMILES mode: {mode}")


def _empty_stats(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "input_values": 0,
        "output_values": 0,
        "invalid_values": 0,
        "multifragment_values": 0,
        "dropped_invalid_values": 0,
        "dropped_multifragment_values": 0,
        "changed_values": 0,
        "duplicate_values_collapsed": 0,
        "unique_before": 0,
        "unique_after": 0,
        "formal_charge_distribution": {},
        "fragment_count_distribution": {},
        "invalid_examples": [],
        "multifragment_examples": [],
    }


def normalize_smiles_values(
    values: Iterable[str | None],
    mode: SmilesMode = "canonical_isomeric",
    *,
    drop_invalid: bool = True,
    drop_multifragment: bool = False,
    deduplicate: bool = True,
    max_examples: int = 10,
) -> tuple[list[str], dict[str, Any]]:
    stats = _empty_stats(mode)
    out: list[str] = []
    seen: set[str] = set()
    before: set[str] = set()
    after: set[str] = set()
    charge_counts: Counter[str] = Counter()
    fragment_counts: Counter[str] = Counter()

    for value in values:
        stats["input_values"] += 1
        cleaned = _clean_smiles(value)
        if cleaned is not None:
            before.add(cleaned)
        profile = profile_smiles(value)
        if not profile.valid:
            stats["invalid_values"] += 1
            if drop_invalid:
                stats["dropped_invalid_values"] += 1
                if len(stats["invalid_examples"]) < max_examples:
                    stats["invalid_examples"].append(value)
                continue
        else:
            charge_counts[str(profile.formal_charge)] += 1
            fragment_counts[str(profile.fragment_count)] += 1
            if profile.multifragment:
                stats["multifragment_values"] += 1
                if len(stats["multifragment_examples"]) < max_examples:
                    stats["multifragment_examples"].append(value)
                if drop_multifragment:
                    stats["dropped_multifragment_values"] += 1
                    continue

        normalized = normalize_smiles(
            value,
            mode,
            drop_invalid=drop_invalid,
            drop_multifragment=drop_multifragment,
        )
        if normalized is None:
            continue
        if cleaned is not None and normalized != cleaned:
            stats["changed_values"] += 1
        if deduplicate and normalized in seen:
            stats["duplicate_values_collapsed"] += 1
            continue
        out.append(normalized)
        seen.add(normalized)
        after.add(normalized)

    stats["output_values"] = len(out)
    stats["unique_before"] = len(before)
    stats["unique_after"] = len(after)
    stats["formal_charge_distribution"] = dict(sorted(charge_counts.items()))
    stats["fragment_count_distribution"] = dict(sorted(fragment_counts.items()))
    return out, stats


def normalize_row_smiles(
    rows: list[MutableMapping[str, Any]],
    mode: SmilesMode = "canonical_isomeric",
    *,
    smiles_col: str = "smiles",
    raw_col: str = "_raw_smiles",
    output_col: str = "smiles",
    drop_invalid: bool = True,
    drop_multifragment: bool = False,
    max_drop_rate: float | None = None,
    max_examples: int = 10,
) -> dict[str, Any]:
    before_rows = len(rows)
    normalized_rows: list[MutableMapping[str, Any]] = []
    values = [row.get(smiles_col) for row in rows]
    _, value_stats = normalize_smiles_values(
        values,
        mode,
        drop_invalid=drop_invalid,
        drop_multifragment=drop_multifragment,
        deduplicate=False,
        max_examples=max_examples,
    )
    dropped_examples: list[dict[str, Any]] = []

    for row in rows:
        original = row.get(smiles_col)
        row[raw_col] = original
        normalized = normalize_smiles(
            original,
            mode,
            drop_invalid=drop_invalid,
            drop_multifragment=drop_multifragment,
        )
        if normalized is None:
            if len(dropped_examples) < max_examples:
                profile = profile_smiles(original)
                reason = profile.error if not profile.valid else "multifragment_smiles"
                dropped_examples.append(
                    {
                        "identifier": row.get("identifier"),
                        "smiles": original,
                        "reason": reason,
                    }
                )
            continue
        profile = profile_smiles(original)
        row[output_col] = normalized
        row["_smiles_canonical_isomeric"] = profile.canonical_isomeric
        row["_smiles_canonical_nonisomeric"] = profile.canonical_nonisomeric
        row["_smiles_inchi_key"] = profile.inchi_key
        row["_smiles_inchi_key_first14"] = profile.inchi_key_first14
        row["_smiles_formula"] = profile.formula
        row["_smiles_formal_charge"] = profile.formal_charge
        row["_smiles_fragment_count"] = profile.fragment_count
        normalized_rows.append(row)

    rows[:] = normalized_rows
    dropped_rows = before_rows - len(rows)
    drop_rate = dropped_rows / before_rows if before_rows else 0.0
    if max_drop_rate is not None and drop_rate > max_drop_rate:
        raise ValueError(
            f"SMILES normalization dropped {drop_rate:.2%} of rows, above max_drop_rate={max_drop_rate:.2%}"
        )

    stats = {
        **value_stats,
        "rows": before_rows,
        "kept_rows": len(rows),
        "dropped_rows": dropped_rows,
        "drop_rate": drop_rate,
        "invalid_rows": value_stats["invalid_values"],
        "multifragment_rows": value_stats["multifragment_values"],
        "changed_rows": value_stats["changed_values"],
        "duplicate_rows_after_normalization": before_rows
        - dropped_rows
        - value_stats["unique_after"],
        "dropped_examples": dropped_examples,
    }
    return stats


def profile_to_dict(profile: SmilesProfile) -> dict[str, Any]:
    return asdict(profile)
