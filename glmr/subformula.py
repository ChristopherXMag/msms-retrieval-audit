"""Candidate-formula-constrained subformula mass decomposition.

The routines in this module deliberately accept candidate compositions and
query mass metadata only.  Ground-truth molecular formulae are not part of the
interface, which makes the resulting features safe for candidate-conditioned
reranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numba import njit
from rdkit import Chem


ELEMENTS = ("H", "C", "O", "N", "P", "S", "Cl", "F", "Br", "I", "B", "As", "Si", "Se")
ATOMIC_NUMBERS = (1, 6, 8, 7, 15, 16, 17, 9, 35, 53, 5, 33, 14, 34)
ATOMIC_NUMBER_TO_INDEX = {value: index for index, value in enumerate(ATOMIC_NUMBERS)}
ELECTRON_MASS = 0.00054858
ROUTE_DIRECT = np.uint8(1)
ROUTE_NEUTRAL_LOSS = np.uint8(2)


def monoisotopic_masses() -> np.ndarray:
    table = Chem.GetPeriodicTable()
    return np.asarray([table.GetMostCommonIsotopeMass(value) for value in ATOMIC_NUMBERS], dtype=np.float64)


def ion_shift(adduct: str) -> float:
    table = Chem.GetPeriodicTable()
    if adduct == "[M+H]+":
        return float(table.GetMostCommonIsotopeMass(1)) - ELECTRON_MASS
    if adduct == "[M+Na]+":
        return float(table.GetMostCommonIsotopeMass(11)) - ELECTRON_MASS
    raise ValueError(f"Unsupported candidate-subformula adduct: {adduct!r}")


def formula_counts_from_smiles(smiles: str) -> tuple[np.ndarray | None, str | None]:
    """Return 14-element counts or an explicit unsupported reason.

    Isotope-labelled and non-CHNOPS/halogen/metalloid molecules intentionally
    fall back to raw E1 rather than receiving an approximate mass assignment.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "rdkit_parse_failed"
    counts = np.zeros(len(ELEMENTS), dtype=np.int16)
    for atom in mol.GetAtoms():
        if atom.GetIsotope():
            return None, "isotope_label"
        atomic_number = int(atom.GetAtomicNum())
        element_index = ATOMIC_NUMBER_TO_INDEX.get(atomic_number)
        if element_index is None:
            return None, f"unsupported_atomic_number_{atomic_number}"
        counts[element_index] += 1
        if atomic_number != 1:
            counts[0] += int(atom.GetTotalNumHs())
    if int(counts.max(initial=0)) > np.iinfo(np.uint8).max:
        return None, "element_count_exceeds_uint8"
    return counts, None


def balanced_partition(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedily balance mixed-radix state counts between two halves."""
    radices = np.asarray(counts, dtype=np.int64) + 1
    ordered = sorted(range(len(radices)), key=lambda index: (-int(radices[index]), index))
    groups: tuple[list[int], list[int]] = ([], [])
    products = [1, 1]
    for index in ordered:
        target = 0 if products[0] <= products[1] else 1
        groups[target].append(index)
        products[target] *= int(radices[index])
    return np.asarray(sorted(groups[0]), dtype=np.int16), np.asarray(sorted(groups[1]), dtype=np.int16)


def enumerate_half(
    counts: np.ndarray, element_indices: np.ndarray, masses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    radices = np.asarray(counts[element_indices], dtype=np.int64) + 1
    total = int(np.prod(radices, dtype=np.int64))
    flat = np.arange(total, dtype=np.int64)
    working = flat.copy()
    digits = np.empty((total, len(element_indices)), dtype=np.uint8)
    state_masses = np.zeros(total, dtype=np.float64)
    for column, (element_index, radix) in enumerate(zip(element_indices.tolist(), radices.tolist())):
        values = working % radix
        working //= radix
        digits[:, column] = values.astype(np.uint8)
        state_masses += values * masses[element_index]
    order = np.argsort(state_masses, kind="stable")
    return state_masses[order], digits[order]


@njit(cache=True)
def _nearest_pairs(
    targets: np.ndarray,
    left_masses: np.ndarray,
    right_masses: np.ndarray,
    ppm_tolerance: float,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find exact top-N bounded-subset pairs for every target mass."""
    rows = len(targets)
    out_left = np.full((rows, top_n), -1, dtype=np.int32)
    out_right = np.full((rows, top_n), -1, dtype=np.int32)
    out_ppm = np.full((rows, top_n), np.inf, dtype=np.float64)
    match_counts = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        target = targets[row]
        if not np.isfinite(target) or target <= 0:
            continue
        tolerance = target * ppm_tolerance * 1e-6
        lower_target = target - tolerance
        upper_target = target + tolerance
        for left_index in range(len(left_masses)):
            low = np.searchsorted(right_masses, lower_target - left_masses[left_index], side="left")
            high = np.searchsorted(right_masses, upper_target - left_masses[left_index], side="right")
            match_counts[row] += high - low
            for right_index in range(low, high):
                theoretical = left_masses[left_index] + right_masses[right_index]
                ppm = abs(theoretical - target) / target * 1e6
                key = left_index * len(right_masses) + right_index
                position = top_n
                for slot in range(top_n):
                    existing_key = out_left[row, slot] * len(right_masses) + out_right[row, slot]
                    if ppm < out_ppm[row, slot] or (ppm == out_ppm[row, slot] and key < existing_key):
                        position = slot
                        break
                if position < top_n:
                    for slot in range(top_n - 1, position, -1):
                        out_ppm[row, slot] = out_ppm[row, slot - 1]
                        out_left[row, slot] = out_left[row, slot - 1]
                        out_right[row, slot] = out_right[row, slot - 1]
                    out_ppm[row, position] = ppm
                    out_left[row, position] = left_index
                    out_right[row, position] = right_index
    return out_left, out_right, out_ppm, match_counts


@dataclass(frozen=True)
class SubformulaMatches:
    counts: np.ndarray
    ppm: np.ndarray
    route: np.ndarray
    ambiguity: np.ndarray
    direct_match_count: np.ndarray
    neutral_loss_match_count: np.ndarray


class BoundedFormulaDecomposer:
    """Reusable exact decomposer for one candidate parent composition."""

    def __init__(self, parent_counts: np.ndarray) -> None:
        counts = np.asarray(parent_counts, dtype=np.int16)
        if counts.shape != (len(ELEMENTS),) or bool((counts < 0).any()):
            raise ValueError(f"Invalid parent composition shape/value: {counts}")
        if int(counts.max(initial=0)) > np.iinfo(np.uint8).max:
            raise ValueError("Parent composition cannot be encoded as uint8")
        self.parent_counts = counts
        self.masses = monoisotopic_masses()
        self.left_elements, self.right_elements = balanced_partition(counts)
        self.left_masses, self.left_digits = enumerate_half(counts, self.left_elements, self.masses)
        self.right_masses, self.right_digits = enumerate_half(counts, self.right_elements, self.masses)

    def _decode(self, left_index: int, right_index: int) -> np.ndarray:
        output = np.zeros(len(ELEMENTS), dtype=np.uint8)
        output[self.left_elements] = self.left_digits[left_index]
        output[self.right_elements] = self.right_digits[right_index]
        return output

    def match(
        self,
        peak_mzs: Iterable[float],
        precursor_mz: float,
        adduct: str,
        ppm_tolerance: float = 20.0,
        top_n: int = 4,
    ) -> SubformulaMatches:
        if top_n < 1:
            raise ValueError("top_n must be positive")
        peaks = np.asarray(list(peak_mzs), dtype=np.float64)
        direct_targets = peaks - ion_shift(adduct)
        loss_targets = float(precursor_mz) - peaks
        d_left, d_right, d_ppm, d_count = _nearest_pairs(
            direct_targets, self.left_masses, self.right_masses, ppm_tolerance, top_n
        )
        l_left, l_right, l_ppm, l_count = _nearest_pairs(
            loss_targets, self.left_masses, self.right_masses, ppm_tolerance, top_n
        )
        output_counts = np.zeros((len(peaks), top_n, len(ELEMENTS)), dtype=np.uint8)
        output_ppm = np.full((len(peaks), top_n), np.inf, dtype=np.float32)
        output_route = np.zeros((len(peaks), top_n), dtype=np.uint8)
        ambiguity = np.minimum(d_count + l_count, np.iinfo(np.uint16).max).astype(np.uint16)
        parent = self.parent_counts.astype(np.int16)
        for peak_index in range(len(peaks)):
            merged: dict[tuple[int, ...], tuple[float, int]] = {}
            for slot in range(top_n):
                if d_left[peak_index, slot] >= 0:
                    composition = self._decode(int(d_left[peak_index, slot]), int(d_right[peak_index, slot]))
                    if int(composition.sum()) > 0:
                        key = tuple(int(value) for value in composition)
                        old = merged.get(key)
                        ppm = float(d_ppm[peak_index, slot])
                        merged[key] = (ppm if old is None else min(ppm, old[0]), int(ROUTE_DIRECT) | (0 if old is None else old[1]))
                if l_left[peak_index, slot] >= 0:
                    loss = self._decode(int(l_left[peak_index, slot]), int(l_right[peak_index, slot])).astype(np.int16)
                    fragment = parent - loss
                    if bool((fragment >= 0).all()) and int(fragment.sum()) > 0:
                        key = tuple(int(value) for value in fragment)
                        old = merged.get(key)
                        ppm = float(l_ppm[peak_index, slot])
                        merged[key] = (
                            ppm if old is None else min(ppm, old[0]),
                            int(ROUTE_NEUTRAL_LOSS) | (0 if old is None else old[1]),
                        )
            ordered = sorted(merged.items(), key=lambda item: (item[1][0], item[0], item[1][1]))[:top_n]
            for slot, (composition, (ppm, route)) in enumerate(ordered):
                output_counts[peak_index, slot] = np.asarray(composition, dtype=np.uint8)
                output_ppm[peak_index, slot] = ppm
                output_route[peak_index, slot] = np.uint8(route)
        return SubformulaMatches(
            counts=output_counts,
            ppm=output_ppm,
            route=output_route,
            ambiguity=ambiguity,
            direct_match_count=np.minimum(d_count, np.iinfo(np.uint16).max).astype(np.uint16),
            neutral_loss_match_count=np.minimum(l_count, np.iinfo(np.uint16).max).astype(np.uint16),
        )
