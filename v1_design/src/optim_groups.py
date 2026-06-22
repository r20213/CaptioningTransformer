from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch.nn as nn


@dataclass
class OptimizerGroups:
    muon: List[nn.Parameter]
    adamw: List[nn.Parameter]
    muon_names: List[str]
    adamw_names: List[str]


def _name_is_in(module_name: str, prefixes: Iterable[str]) -> bool:
    return any(module_name.startswith(prefix) for prefix in prefixes)


def _is_embedding_or_norm(module: nn.Module) -> bool:
    return isinstance(module, (nn.Embedding, nn.LayerNorm)) or module.__class__.__name__.lower().endswith("norm")


def build_optimizer_groups(model: nn.Module) -> OptimizerGroups:
    module_map: Dict[str, nn.Module] = dict(model.named_modules())

    muon_params: List[nn.Parameter] = []
    adamw_params: List[nn.Parameter] = []
    muon_names: List[str] = []
    adamw_names: List[str] = []

    for full_name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module_name, _, param_name = full_name.rpartition(".")
        parent_module = module_map.get(module_name, None)

        force_adamw = False
        if parent_module is not None and _is_embedding_or_norm(parent_module):
            force_adamw = True
        if "visual_projector" in full_name:
            force_adamw = True
        if param_name == "bias":
            force_adamw = True

        if not force_adamw and param.ndim >= 2:
            muon_params.append(param)
            muon_names.append(full_name)
        else:
            adamw_params.append(param)
            adamw_names.append(full_name)

    return OptimizerGroups(
        muon=muon_params,
        adamw=adamw_params,
        muon_names=muon_names,
        adamw_names=adamw_names,
    )


def count_params(params: List[nn.Parameter]) -> int:
    return int(sum(p.numel() for p in params))


def summarize_groups(groups: OptimizerGroups) -> str:
    muon_count = count_params(groups.muon)
    adamw_count = count_params(groups.adamw)
    total = muon_count + adamw_count

    lines = [
        "Optimizer Group Summary",
        f"- Muon params: {muon_count:,}",
        f"- AdamW params: {adamw_count:,}",
        f"- Total params in optimizer groups: {total:,}",
        f"- Muon tensors: {len(groups.muon)}",
        f"- AdamW tensors: {len(groups.adamw)}",
    ]
    return "\n".join(lines)
