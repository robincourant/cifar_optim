import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class BinaryQuantizer:
    def __init__(self, net):
        # super(BinaryQuantizer, self).__init__()
        self.quantizer_name = "binary"
        self.name = "_".join([net.name, self.quantizer_name])
        self.net = net

        n_conv_linear = 0
        for m in self.net.modules():
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

        index = -1
        for m in self.net.modules():
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

        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                self.target_modules[index].data.sign()
            )
            self.target_modules[index].int()

    def restore_params(self):
        """Restore previous model's target modules."""
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip_params(self):
        """Clip all parameters to the range [-1,1]."""
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                self.clipper(Variable(self.target_modules[index].data)).data
            )

    def get_bool_params(self):
        """Transform -1,1 weights in False,True."""
        for index in range(self.num_of_params):
            self.target_modules[index].data = (
                self.target_modules[index].data > 0
            )
            self.target_modules[index].requires_grad = False

    def get_float_params(self):
        """Transform False-True weights in -1,1."""
        for index in range(self.num_of_params):
            self.target_modules[index].data = (
                self.target_modules[index].data.float() * 2 - 1
            )
            self.target_modules[index].requires_grad = True

    def forward(self, x):
        """Perform the forward propagation given a sample."""
        x = self.net(x)
        return x


class HalfQuantizer:
    def __init__(self, net):
        self.quantizer_name = "half"
        self.name = "_".join([net.name, self.quantizer_name])
        self.net = net.half()

    def forward(self, x):
        """Perform the forward propagation given a sample."""
        x = self.net(x.half())
        return x
