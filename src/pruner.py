from typing import Callable, List, Tuple

import torch
from torch import nn
import torch.nn.utils.prune as prune

from learner import Learner


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


class GlobalPruner(Pruner):
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
        super(GlobalPruner, self).__init__(learner)
        self.parameters_to_prune = (
            parameters_to_prune
            if parameters_to_prune
            else self._get_all_parameters()
        )

    def prune_model(self, pruning_rate: float = 0.2):
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
        for module in self.net.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                parameters_to_prune.append((module, "weight"))

        return tuple(parameters_to_prune)


class FilterPruner(Pruner):
    def __init__(self, learner: Learner, n_filters: int):
        super(FilterPruner, self).__init__(learner)
        self.n_filters = n_filters

    @staticmethod
    def _get_relevant_filters(
        kernel: torch.Tensor, n_filters: int
    ) -> List[int]:
        """
        Find the filters to keep after the pruning by detecting the `n_filters`
        filters with the lowest L1 norm.
        """
        kernel_norms = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), 1)
        _, indices = torch.sort(kernel_norms, descending=True)
        n_kept_filters = indices.shape[0] - n_filters
        indices_to_keep = indices[:n_kept_filters].tolist()

        return indices_to_keep

    @staticmethod
    def _prune_conv(
        conv: nn.Module,
        indices_to_keep: List[int],
        io_dim: int,
    ) -> nn.Module:
        """"""
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
        import ipdb

        ipdb.set_trace()
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
        """"""
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
        """"""
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

    def _replace_layer(
        self,
        prune_method: Callable,
        layer_name: str,
        layer_params: nn.Module,
        indices_to_keep: List[int],
        prune_input: int,
    ):
        """"""
        # Build a new layer without the input pruned filters
        pruned_layer = self.replace_method(
            layer_params,
            indices_to_keep,
            prune_input,
        )
        # Replace the layer by the pruned layer
        setattr(self.net, layer_name, pruned_layer)

    def prune_model(self):
        indices_to_keep = []
        prune_input = 0  # Wether to prune input or output of the layer
        for layer_name, original_layer in list(self.net.named_modules())[1:]:
            # Handle convolutional layers
            if isinstance(original_layer, nn.Conv2d):
                # Modify the input of the layer
                if prune_input:
                    # Build a new layer without the input pruned filters
                    pruned_layer = self._prune_conv(
                        original_layer,
                        indices_to_keep,
                        prune_input,
                    )
                    # Replace the layer by the pruned layer
                    setattr(self.net, layer_name, pruned_layer)
                    original_layer = pruned_layer
                    prune_input ^= 1

                # Modify the output of the layer
                if not prune_input:
                    indices_to_keep = self._get_relevant_filters(
                        original_layer.weight.data, self.n_filters
                    )
                    # Build a new layer without the output pruned filters
                    pruned_layer = self._prune_conv(
                        original_layer,
                        indices_to_keep,
                        prune_input,
                    )
                    # Replace the layer by the pruned layer
                    setattr(self.net, layer_name, pruned_layer)
                    original_layer = pruned_layer
                    prune_input ^= 1

            # Handle batch norm layers (only modify input)
            if isinstance(original_layer, nn.BatchNorm2d) and prune_input:
                # Build a new layer without the output pruned filters
                pruned_layer = self._prune_norm(
                    original_layer,
                    indices_to_keep,
                    prune_input,
                )
                setattr(self.net, layer_name, pruned_layer)
                original_layer = pruned_layer
                prune_input ^= 1

            # Handle linear layers (only modify input)
            if isinstance(original_layer, nn.Linear) and prune_input:
                # Build a new layer without the output pruned filters
                pruned_layer = self._prune_linear(
                    original_layer,
                    indices_to_keep,
                    prune_input,
                )
                setattr(self.net, layer_name, pruned_layer)
                original_layer = pruned_layer
                prune_input ^= 1
