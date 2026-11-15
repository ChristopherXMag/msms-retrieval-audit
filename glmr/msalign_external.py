"""Released MIT MSAlign port for frozen E1; no Lightning/test lifecycle.

Copyright (c) 2026 KrzakalaPaul. Full notice: docs/msalign/third_party/MSAlign_LICENSE.txt.
Source pin c2ef425b68749052a17b9cdd5f5f11709322654e.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import torch
from torch import nn
import torch.nn.functional as F

ADDUCT_ALIASES = {
    "[M+CH3COO]-": "[M+CH3COOH-H]-",
    "[M+FA-H]-": "[M+HCOOH-H]-",
    "[M+HCOO]-": "[M+HCOOH-H]-",
}


def normalize_adduct(adduct):
    """Map equivalent adduct spellings to the same category."""

    if not isinstance(adduct, str):
        return adduct
    compact = adduct.replace(" ", "")
    return ADDUCT_ALIASES.get(compact, compact)


class CollisionEnergyEncoder(nn.Module):
    r"""Sinusoidal encoding of collision energy.

    For energy :math:`e`, dimensions ``2i`` and ``2i+1`` contain
    :math:`\sin(e/(100\,10000^{2i/d}))` and the corresponding cosine. Missing
    energies use a learned vector.
    """

    def __init__(self, d_out: int) -> None:
        super().__init__()
        if d_out % 2:
            raise ValueError("The collision-energy dimension must be even")
        positions = torch.arange(0, d_out, 2, dtype=torch.float32)
        self.register_buffer("frequencies", 10_000.0 ** (positions / d_out))
        self.missing = nn.Parameter(torch.empty(d_out))
        nn.init.normal_(self.missing, std=0.02)

    def forward(self, energy: torch.Tensor) -> torch.Tensor:
        missing = torch.isnan(energy)
        angles = energy.nan_to_num().unsqueeze(-1) / 100.0 / self.frequencies
        encoded = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        return torch.where(missing.unsqueeze(-1), self.missing, encoded)


class AdductEncoder(nn.Module):
    """Learn adduct identity and charge-sign embeddings.

    Half the output represents the category and half represents its sign.
    Unknown or missing adducts have their own category and sign entries.
    """

    def __init__(self, n_adducts: int, d_out: int, unknown_index: int, adduct_signs: Sequence[int]):
        super().__init__()
        if d_out % 2:
            raise ValueError("The adduct dimension must be even")
        self.category = nn.Embedding(n_adducts, d_out // 2)
        self.sign = nn.Embedding(3, d_out // 2)  # negative, positive, unknown
        nn.init.normal_(self.category.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.sign.weight, mean=0.0, std=0.02)
        signs = torch.as_tensor(adduct_signs)
        sign_indices = torch.where(signs == 0, 2, (signs + 1) // 2)
        self.register_buffer("sign_indices", sign_indices.long())
        self.unknown_index = int(unknown_index)

    def forward(self, adduct: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.category(adduct), self.sign(self.sign_indices[adduct])), dim=-1)


class MLP(nn.Module):
    """An MLP with ``n_hidden_layers`` hidden blocks and one output projection.

    A hidden block is ``Linear -> LayerNorm -> GELU -> Dropout``. Consequently,
    ``n_hidden_layers=0`` is exactly one linear projection.
    """

    def __init__(self, d_in, d_hidden, d_out, n_hidden_layers, dropout):
        super().__init__()
        layers: list[nn.Module] = []
        width = d_in
        for _ in range(n_hidden_layers):
            layers.extend(
                (nn.Linear(width, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout))
            )
            width = d_hidden
        layers.append(nn.Linear(width, d_out))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MSAlign(nn.Module):
    """Align one spectrum representation with one molecular representation.

    ``input_dimensions`` is inferred by the data module. Keeping it separate from
    YAML lets the same architecture consume DreaMS or bins and MolDeBERTa or
    Morgan fingerprints without representation-specific model code.
    """

    def __init__(self, config: Mapping, input_dimensions: Mapping) -> None:
        super().__init__()
        self.config = dict(config)
        model = config["model"]
        training = config["training"]

        d_shared = int(model["d_shared"])
        d_hidden = int(model["d_hidden"])
        d_metadata = int(model["d_metadata"])
        dropout = float(model["dropout"])
        use_energy = bool(model.get("use_collision_energy", True))
        use_adduct = bool(model.get("use_adduct", True))
        n_metadata = int(use_energy) + int(use_adduct)
        if n_metadata and d_metadata % n_metadata:
            raise ValueError("d_metadata must be divisible by the number of metadata fields")
        d_metadata_part = d_metadata // n_metadata if n_metadata else 0

        self.ms_pre = MLP(
            int(input_dimensions["d_ms"]), d_hidden, d_shared,
            int(model["n_hidden_layers_ms_pre"]), dropout,
        )
        self.energy_encoder = CollisionEnergyEncoder(d_metadata_part) if use_energy else None
        self.adduct_encoder = (
            AdductEncoder(
                int(input_dimensions["n_adducts"]), d_metadata_part,
                int(input_dimensions["unknown_adduct_index"]),
                input_dimensions["adduct_signs"],
            )
            if use_adduct else None
        )
        self.ms_post = MLP(
            d_shared + (d_metadata if n_metadata else 0), d_hidden, d_shared,
            int(model["n_hidden_layers_ms_post"]), dropout,
        )
        self.mol = MLP(
            int(input_dimensions["d_mol"]), d_hidden, d_shared,
            int(model["n_hidden_layers_mol"]), dropout,
        )

        # Optimizing log(tau) keeps the temperature positive without clipping.
        self.log_temperature = nn.Parameter(
            torch.tensor(float(model.get("temperature_init", 0.07))).log()
        )
        self.lr = float(training["learning_rate"])
        self.weight_decay = float(training["weight_decay"])
        self.n_warmup_steps = int(training["n_warmup_steps"])
        self.n_max_steps = int(training["n_max_steps"])

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def encode_spectrum(
        self, spectrum: torch.Tensor, metadata: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return the unnormalized spectrum vector in the shared space."""

        spectrum = self.ms_pre(spectrum.float())
        metadata_vectors = []
        if self.energy_encoder is not None:
            metadata_vectors.append(self.energy_encoder(metadata["collision_energy"]))
        if self.adduct_encoder is not None:
            metadata_vectors.append(self.adduct_encoder(metadata["adduct"]))
        if metadata_vectors:
            spectrum = torch.cat((spectrum, *metadata_vectors), dim=-1)
        return self.ms_post(spectrum)

    def encode_molecule(self, molecule: torch.Tensor) -> torch.Tensor:
        """Return the unnormalized molecule vector in the shared space."""

        return self.mol(molecule.float())

    def forward(
        self,
        spectrum: torch.Tensor,
        candidates: torch.Tensor,
        metadata: Mapping[str, torch.Tensor],
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates; candidate zero is the target by construction."""

        spectrum = F.normalize(self.encode_spectrum(spectrum, metadata), dim=-1)
        valid_indices = candidate_mask.nonzero(as_tuple=False)
        molecules = F.normalize(self.encode_molecule(candidates[candidate_mask]), dim=-1)
        scores = spectrum.new_zeros(candidate_mask.shape)
        scores[candidate_mask] = (
            spectrum[valid_indices[:, 0]] * molecules
        ).sum(dim=-1) / self.temperature
        return scores.masked_fill(~candidate_mask, float("-inf"))
