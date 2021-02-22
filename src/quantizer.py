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
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                self.target_modules[index].data > 0
            )

    def get_float_params(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                self.target_modules[index].data * 2 - 1
            )

    def forward(self, x):
        """Perform the forward propagation given a sample."""
        x = self.net(x)
        return x


class HalfQuantizer:
    def __init__(self, net):
        self.quantizer_name = "half"
        self.name = "_".join([net.name, self.quantizer_name])
        self._net = net.half()

    def forward(self, x):
        """Perform the forward propagation given a sample."""
        x = self._net(x.half())
        return x


class BC:
    def __init__(self, model):

        # First we need to
        # count the number of Conv2d and Linear
        #  This will be used next in order to build a list of all
        # parameters of the model

        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets - 1
        self.bin_range = (
            np.linspace(start_range, end_range, end_range - start_range + 1)
            .astype("int")
            .tolist()
        )

        #  Now we can initialize the list of parameters

        self.num_of_params = len(self.bin_range)
        self.saved_params = (
            []
        )  #  This will be used to save the full precision weights

        self.target_modules = (
            []
        )  #  this will contain the list of modules to be modified

        self.net = model  # this contains the model that will be trained and quantified

        ### This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def save_params(self):

        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters

        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarize_params(self):

        ### To be completed

        ### (1) Save the current full precision parameters using the save_params method
        self.save_params()

        ### (2) Binarize the weights in the model, by iterating through the list of target modules and overwrite the values with their binary version
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                self.target_modules[index].data.sign()
            )

    def restore_params(self):

        ### restore the copy from self.saved_params into the model

        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip_params(self):

        clip_scale = []
        m = nn.Hardtanh(-1, 1)
        for index in range(self.num_of_params):
            clip_scale.append(m(Variable(self.target_modules[index].data)))
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(clip_scale[index].data)
        ## To be completed
        ## Clip all parameters to the range [-1,1] using Hard Tanh
        ## you can use the nn.Hardtanh function

    def forward(self, x):
        ### This function is used so that the model can be used while training
        out = self.net(x)
        return out
