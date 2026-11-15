from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from rdkit import Chem  # type: ignore
except ImportError:  # pragma: no cover
    Chem = None


def rdkit_connectivity(smiles: str, fallback: str) -> str:
    if Chem is None:
        raise RuntimeError("RDKit is required for connectivity-safe counterfactual pairing")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return fallback
    try:
        return str(Chem.MolToInchiKey(molecule)).split("-", 1)[0]
    except Exception:
        return fallback


def _stable_select(
    values: list[int], count: int, seed: int, records: Sequence[dict[str, Any]]
) -> list[int]:
    def key(index: int) -> str:
        identifier = str(records[index].get("identifier") or index)
        return hashlib.sha256(f"{seed}|{identifier}".encode()).hexdigest()

    return sorted(values, key=key)[:count]


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def build_counterfactual_donor_assignment(
    records: Sequence[dict[str, Any]],
    *,
    mass_round_decimals: int,
    seed: int,
    match_instrument: bool = True,
    match_ce_known: bool = True,
    jitter: float = 0.02,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Build a closed, one-to-one spectrum derangement in strict metadata strata.

    Dominant connectivity groups are deterministically trimmed so that a feasible
    cross-connectivity assignment exists. The Hungarian cost matches peak-count
    density; it never relaxes connectivity or metadata constraints.
    """

    identity_cache: dict[str, str] = {}
    identities: list[str] = []
    stored_mismatches = 0
    strata: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        smiles = str(record.get("canonical_isomeric_smiles") or "")
        stored = str(record.get("inchikey") or "").split("-", 1)[0]
        if smiles not in identity_cache:
            identity_cache[smiles] = rdkit_connectivity(smiles, stored or smiles)
        identity = identity_cache[smiles]
        identities.append(identity)
        stored_mismatches += int(bool(stored) and stored != identity)

        key: list[Any] = [
            str(record.get("adduct") or ""),
            round(float(record["parent_mass"]), mass_round_decimals),
        ]
        if match_instrument:
            key.append(str(record.get("instrument_type") or ""))
        if match_ce_known:
            key.append(record.get("collision_energy") is not None)
        strata[tuple(key)].append(index)

    selected_strata: list[list[int]] = []
    raw_eligible_rows = 0
    trimmed_rows = 0
    for key in sorted(strata):
        rows = strata[key]
        by_identity: dict[str, list[int]] = defaultdict(list)
        for index in rows:
            by_identity[identities[index]].append(index)
        if len(by_identity) < 2:
            continue
        raw_eligible_rows += len(rows)
        dominant_identity, dominant_rows = max(
            by_identity.items(), key=lambda item: (len(item[1]), item[0])
        )
        other_count = len(rows) - len(dominant_rows)
        if len(dominant_rows) > other_count:
            keep = _stable_select(dominant_rows, other_count, seed, records)
            selected = keep + [
                index
                for identity, indices in by_identity.items()
                if identity != dominant_identity
                for index in indices
            ]
            trimmed_rows += len(dominant_rows) - len(keep)
        else:
            selected = list(rows)
        if len(selected) >= 2:
            selected_strata.append(sorted(selected))

    rng = np.random.default_rng(seed)
    assignment: dict[int, int] = {}
    for rows in selected_strata:
        peaks = np.log1p(
            np.asarray([float(records[index].get("num_peaks") or 0.0) for index in rows])
        )
        cost = np.abs(peaks[:, None] - peaks[None, :])
        if jitter > 0:
            cost += rng.uniform(0.0, jitter, size=cost.shape)
        identity_values = np.asarray([identities[index] for index in rows], dtype=object)
        cost[identity_values[:, None] == identity_values[None, :]] = 1e9
        recipient_pos, donor_pos = linear_sum_assignment(cost)
        if not np.array_equal(recipient_pos, np.arange(len(rows))):
            raise RuntimeError("Counterfactual assignment did not cover every recipient")
        for left, right in zip(recipient_pos.tolist(), donor_pos.tolist(), strict=True):
            recipient = rows[left]
            donor = rows[right]
            if identities[recipient] == identities[donor]:
                raise RuntimeError("Counterfactual assignment violated connectivity derangement")
            assignment[recipient] = donor

    recipients = sorted(assignment)
    donors = [assignment[index] for index in recipients]
    mass_ppm = np.asarray(
        [
            abs(float(records[left]["parent_mass"]) - float(records[right]["parent_mass"]))
            / max(abs(float(records[left]["parent_mass"])), 1e-12)
            * 1e6
            for left, right in zip(recipients, donors, strict=True)
        ]
    )
    peak_delta = np.asarray(
        [
            abs(
                math.log1p(float(records[left].get("num_peaks") or 0.0))
                - math.log1p(float(records[right].get("num_peaks") or 0.0))
            )
            for left, right in zip(recipients, donors, strict=True)
        ]
    )
    audit = {
        "records": len(records),
        "recipients": len(recipients),
        "recipient_fraction": len(recipients) / max(len(records), 1),
        "unique_donors": len(set(donors)),
        "one_to_one": len(set(donors)) == len(donors),
        "closed_permutation": set(recipients) == set(donors),
        "fixed_points": sum(left == right for left, right in zip(recipients, donors, strict=True)),
        "same_connectivity_violations": sum(
            identities[left] == identities[right]
            for left, right in zip(recipients, donors, strict=True)
        ),
        "same_adduct_violations": sum(
            str(records[left].get("adduct") or "")
            != str(records[right].get("adduct") or "")
            for left, right in zip(recipients, donors, strict=True)
        ),
        "same_instrument_violations": sum(
            str(records[left].get("instrument_type") or "")
            != str(records[right].get("instrument_type") or "")
            for left, right in zip(recipients, donors, strict=True)
        ),
        "same_ce_known_violations": sum(
            (records[left].get("collision_energy") is None)
            != (records[right].get("collision_energy") is None)
            for left, right in zip(recipients, donors, strict=True)
        ),
        "mass_round_decimals": mass_round_decimals,
        "match_instrument": match_instrument,
        "match_ce_known": match_ce_known,
        "multi_connectivity_strata": len(selected_strata),
        "raw_rows_in_multi_connectivity_strata": raw_eligible_rows,
        "dominant_identity_rows_trimmed": trimmed_rows,
        "parent_mass_delta_ppm": _quantiles(mass_ppm),
        "abs_log1p_peak_count_delta": _quantiles(peak_delta),
        "rdkit_connectivity_groups": len(set(identities)),
        "stored_inchikey_mismatch_rows": stored_mismatches,
        "seed": seed,
        "jitter": jitter,
    }
    if any(
        audit[key]
        for key in (
            "fixed_points",
            "same_connectivity_violations",
            "same_adduct_violations",
            "same_instrument_violations" if match_instrument else "fixed_points",
            "same_ce_known_violations" if match_ce_known else "fixed_points",
        )
    ):
        raise RuntimeError(f"Invalid counterfactual assignment: {audit}")
    return assignment, audit
