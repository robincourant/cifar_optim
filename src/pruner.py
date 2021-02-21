import functools
from typing import Dict, List

import torch
from torch import nn
import torch.nn.utils.prune as prune

from learner import Learner
from models import PreActBlock


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


class Pruner:
    def __init__(
        self,
        learner: Learner,
    ):
        self.learner = learner
        self.net = self.learner.net

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


class UnstructuredPruner(Pruner):
    """
    Perform a pruning by setting to zero the lowest x% of connections across
    the given layers to prune.
    """

    def __init__(
        self,
        learner: Learner,
    ):
        super(UnstructuredPruner, self).__init__(learner)

    def prune_model(self, layer_pruning_rates: Dict[str, float]):
        """Prune parameters by setting to 0 the lowest weights (L1 norm).

        :param layer_pruning_rates: pruning rates with each layer's names.
        """
        for layer_name, pruning_rate in layer_pruning_rates.items():
            pruning_parameters = self._get_pruning_parameters(layer_name)
            prune.global_unstructured(
                pruning_parameters,
                pruning_method=prune.L1Unstructured,
                amount=pruning_rate,
            )

    def _get_pruning_parameters(self, layer_name: str):
        """List all layers to be pruned."""
        layers_to_prune = []
        for m in self.net._modules[layer_name].modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                layers_to_prune.append((m, "weight"))

        return tuple(layers_to_prune)


class StructuredPruner(Pruner):
    """
    Perform a pruning by removing the lowest x% of filters across
    the given layers to prune.
    """

    def __init__(self, learner: Learner):
        super(StructuredPruner, self).__init__(learner)

    @staticmethod
    def _get_relevant_filters(
        kernel: torch.Tensor, pruning_rate: float
    ) -> List[int]:
        """
        Find the filters to keep after the pruning by finding the
        `pruning_rate`% of the module filters with the lowest L1 norm.
        """
        kernel_norms = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), 1)
        _, indices = torch.sort(kernel_norms, descending=True)
        n_filters_to_remove = int(pruning_rate * len(indices))
        n_filters_to_keep = indices.shape[0] - n_filters_to_remove
        indices_to_keep = indices[:n_filters_to_keep].tolist()

        return indices_to_keep

    @staticmethod
    def _prune_conv(
        conv: nn.Module,
        indices_to_keep: List[int],
        io_dim: int,
    ) -> nn.Module:
        """
        Create a new convolutional layer from an original one but remove
        unwanted filters.
        """
        in_channels = len(indices_to_keep) if io_dim else conv.in_channels
        out_channels = conv.out_channels if io_dim else len(indices_to_keep)

        pruned_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
        )
        # Update weight and bias of the pruned layer
        pruned_conv.weight.data = torch.index_select(
            conv.weight.data, io_dim, torch.tensor(indices_to_keep)
        )
        # The bias is not modified if it is the input which is modified
        if conv.bias is not None:
            if io_dim:
                pruned_conv.bias.data = conv.bias.data
            else:
                pruned_conv.bias.data = torch.index_select(
                    conv.bias.data, io_dim, torch.tensor(indices_to_keep)
                )
        else:
            pruned_conv.bias = conv.bias

        return pruned_conv

    @staticmethod
    def _prune_norm(
        norm: nn.Module,
        indices_to_keep: List[int],
        io_dim: int,
    ) -> nn.Module:
        """
        Create a new batch norm layer from an original one but remove
        unwanted connections.
        """
        pruned_norm = torch.nn.BatchNorm2d(
            num_features=len(indices_to_keep),
            eps=norm.eps,
            momentum=norm.momentum,
            affine=norm.affine,
            track_running_stats=norm.track_running_stats,
        )
        pruned_norm.weight.data = torch.index_select(
            norm.weight.data, 0, torch.tensor(indices_to_keep)
        )
        pruned_norm.bias.data = torch.index_select(
            norm.bias.data, 0, torch.tensor(indices_to_keep)
        )

        if norm.track_running_stats:
            pruned_norm.running_mean.data = torch.index_select(
                norm.running_mean.data, 0, torch.tensor(indices_to_keep)
            )
            pruned_norm.running_var.data = torch.index_select(
                norm.running_var.data, 0, torch.tensor(indices_to_keep)
            )

        return pruned_norm

    @staticmethod
    def _prune_linear(
        linear: nn.Module,
        indices_to_keep: List[int],
        io_dim: int,
    ) -> nn.Module:
        """
        Create a new linear layer from an original one but remove
        unwanted connections.
        """
        pruned_linear = torch.nn.Linear(
            in_features=len(indices_to_keep),
            out_features=linear.out_features,
            bias=linear.bias is not None,
        )
        pruned_linear.weight.data = torch.index_select(
            linear.weight.data, io_dim, torch.tensor(indices_to_keep)
        )
        pruned_linear.bias.data = linear.bias.data

        return pruned_linear

    def prune_model(self, layer_pruning_rates: Dict[str, float]):
        """Prune parameters by removing the lowest filters (L1 norm).

        :param layer_pruning_rates: pruning rates with each layer's names.
        """
        indices_to_keep = []
        first_linear = 1  # Wether the first fc layer has already been met
        prune_input = 0  # Wether to prune input or output of the layer
        _, pruning_rate = sorted(layer_pruning_rates.items())[0]
        for layer_name, original_layer in list(self.net.named_modules())[1:]:
            if layer_name in layer_pruning_rates:
                pruning_rate = layer_pruning_rates[layer_name]

            # Handle blocks with shortcut
            if isinstance(original_layer, PreActBlock) and hasattr(
                original_layer, "shortcut"
            ):
                # Save input filters to keep for next shortcut
                shortcut_inputs_to_keep = indices_to_keep

            # Handle convolutional layers that are not shortcuts
            if isinstance(original_layer, nn.Conv2d) and (
                "shortcut" not in layer_name
            ):
                # Modify the input of the layer
                if prune_input:
                    # Build a new layer without the input pruned filters
                    pruned_layer = self._prune_conv(
                        original_layer,
                        indices_to_keep,
                        prune_input,
                    )
                    # Replace the layer by the pruned layer
                    rsetattr(self.net, layer_name, pruned_layer)
                    original_layer = pruned_layer
                    prune_input ^= 1

                # Modify the output of the layer
                if not prune_input:
                    indices_to_keep = self._get_relevant_filters(
                        original_layer.weight.data, pruning_rate
                    )
                    # Build a new layer without the output pruned filters
                    pruned_layer = self._prune_conv(
                        original_layer,
                        indices_to_keep,
                        prune_input,
                    )
                    # Replace the layer by the pruned layer
                    rsetattr(self.net, layer_name, pruned_layer)
                    original_layer = pruned_layer
                    shortcut_output_to_keep = indices_to_keep
                    prune_input ^= 1

            # Handle convolutional layers that are shortcuts
            if isinstance(original_layer, nn.Conv2d) and (
                "shortcut" in layer_name
            ):
                # Build a new layer without the input pruned filters
                input_pruned_layer = self._prune_conv(
                    original_layer,
                    shortcut_inputs_to_keep,
                    1,
                )

                # Build a new layer without the output pruned filters
                pruned_shortcut = self._prune_conv(
                    input_pruned_layer,
                    shortcut_output_to_keep,
                    0,
                )
                # Replace the layer by the pruned layer
                rsetattr(self.net, layer_name, pruned_shortcut)

            # Handle batch norm layers (only modify input)
            if isinstance(original_layer, nn.BatchNorm2d) and prune_input:
                # Build a new layer without the output pruned filters
                pruned_layer = self._prune_norm(
                    original_layer,
                    indices_to_keep,
                    prune_input,
                )
                rsetattr(self.net, layer_name, pruned_layer)
                original_layer = pruned_layer

            # Handle linear layers (only modify input)
            if isinstance(original_layer, nn.Linear) and prune_input:
                if first_linear:
                    # Build a new layer without the output pruned filters
                    pruned_layer = self._prune_linear(
                        original_layer,
                        indices_to_keep,
                        prune_input,
                    )
                    rsetattr(self.net, layer_name, pruned_layer)
                    original_layer = pruned_layer
                    first_linear ^= 1
