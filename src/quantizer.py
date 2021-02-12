import torch.nn as nn
import numpy as np


class BinaryQuantizer(nn.Module):
    def __init__(self, net):
        super(BinaryQuantizer, self).__init__()
        self.quantizer = "binary"
        self.name = "_".join([net.name, self.quantizer])
        n_conv_linear = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                n_conv_linear += 1
        self.bin_range = (
            np.linspace(0, n_conv_linear - 1, n_conv_linear)
            .astype("int")
            .tolist()
        )
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []

        self._net = net

        index = -1
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index += 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

        self.clipper = nn.Hardtanh()

    def save_params(self):
        """Save current model's target modules."""
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarize_params(self):
        """Binarize the weights in the net."""
        # Save the current full precision parameters
        self.save_params()

        binarized_target_modules = []
        for target_module in self.target_modules:
            binarized_target_modules.append((target_module >= 0) * 2 - 1)
        self.target_modules = binarized_target_modules

    def restore_params(self):
        """Restore previous model's target modules."""
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip_params(self):
        """Clip all parameters to the range [-1,1]."""
        clipped_target_modules = []
        for target_module in self.target_modules:
            clipped_target_modules.append(self.clipper(target_module))
        self.target_modules = clipped_target_modules

    def forward(self, x):
        """Perform the forward propagation given a sample."""
        x = self._net(x)
        return x


class HalfQuantizer(nn.Module):
    def __init__(self, net):
        super(HalfQuantizer, self).__init__()
        self.quantizer = "half"
        self.name = "_".join([net.name, self.quantizer])
        self._net = net.half()

    def forward(self, x):
        """Perform the forward propagation given a sample."""
        x = self._net(x.half())
        return x
