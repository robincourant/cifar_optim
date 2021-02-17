from typing import Tuple

import torch
from torch import nn
import torch.nn.utils.prune as prune

from learner import Learner


class GlobalPruner:
    """
    Perform a global pruning by removing the lowest x% of connections across
    the whole model.
    """

    def __init__(
        self,
        learner: Learner,
        parameters_to_prune: Tuple[Tuple[nn.Module, str]] = None,
    ):
        """
        :param learner: learner to prune.
        :param parameters_to_prune: layers to prune, if None, then prune the
            entire model.
        """
        self.learner = learner
        self.net = self.learner.net
        self.parameters_to_prune = (
            parameters_to_prune
            if parameters_to_prune
            else self._get_all_parameters()
        )

    def prune_parameters(self, pruning_rate: float = 0.2):
        """Prune parameters using the L1 norm as a measure.

        :param pruning_rate: percentage of connections to prune (x%).
        """
        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=pruning_rate,
        )

    def _get_all_parameters(self):
        """List all layer to be pruned."""
        parameters_to_prune = []
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                parameters_to_prune.append((m, "weight"))

        return tuple(parameters_to_prune)

    def get_sparsity(self):
        """Print sparsity of each layer and model global sparsity."""
        global_null_weight = 0
        global_total_weight = 0
        for layer_name, layer_params in list(self.net.named_modules())[1:]:
            if isinstance(layer_params, nn.Conv2d) or isinstance(
                layer_params, nn.Linear
            ):
                layer_null_weight = float(torch.sum(layer_params.weight == 0))
                layer_total_weight = float(layer_params.weight.nelement())
                layer_sparsity = layer_null_weight / layer_total_weight
                global_null_weight += layer_null_weight
                global_total_weight += layer_total_weight

                print(f"Sparsity of {layer_name}: {layer_sparsity:.2%}")

        global_sparsity = global_null_weight / global_total_weight
        print(f"Global sparsity: {global_sparsity:.2%}")
