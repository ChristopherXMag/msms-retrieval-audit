"""Load pinned GLACIER math without unused Ray/hyperparameter-search imports.

No model functions are stubbed. nn_utils' eager optional search re-exports are
omitted; the existing functional-SDPA eval dropout correction is mandatory.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def load_glacier(ms_pred_root: Path):
    source = ms_pred_root.resolve() / "src"
    if not (source / "ms_pred/glacier/joint_model.py").is_file():
        raise FileNotFoundError(source)
    sys.path.insert(0, str(source))
    import ms_pred
    if "ms_pred.nn_utils" not in sys.modules:
        namespace = types.ModuleType("ms_pred.nn_utils")
        namespace.__path__ = [str(source / "ms_pred/nn_utils")]
        namespace.__package__ = "ms_pred.nn_utils"
        sys.modules[namespace.__name__] = namespace
        ms_pred.nn_utils = namespace
        # Common's misc_utils refers to nn_utils only inside function bodies.
        importlib.import_module("ms_pred.common")
        for module in ("nn_utils", "mol_graph", "transformer_layer", "form_embedder", "dgl_graph_ops"):
            actual = importlib.import_module(f"ms_pred.nn_utils.{module}")
            for name, value in vars(actual).items():
                if not name.startswith("_"):
                    setattr(namespace, name, value)
    common = importlib.import_module("ms_pred.common")
    common.instrument2onehot_pos = {"Orbitrap": 0, "QTOF": 1}
    dataset = importlib.import_module("ms_pred.glacier.dataset")
    model = importlib.import_module("ms_pred.glacier.joint_model")
    from tools.generation.run_glacier_seeded_predict import patch_graphormer_eval_dropout
    patch_graphormer_eval_dropout()
    return common, dataset.TreeProcessor, dataset.IntenDataset, model.JointModel
