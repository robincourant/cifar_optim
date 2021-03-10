from functools import partial

import torch
import torch.nn as nn

from src.models import PreActResNet


def count_conv2d(m, x, y, r):
    x = x[0]  # remove tuple

    fin = m.in_channels
    sh, sw = m.kernel_size

    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    kernel_mul = kernel_mul * r
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0]  # remove tuple

    nelements = x.numel()
    total_sub = 2 * nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_linear(m, x, y, r):
    # per output element
    total_mul = m.in_features * r
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    m.total_ops += torch.Tensor([int(total_ops)])


def count_sequential(m, x, y):
    pass


# custom ops could be used to pass variable customized ratios for quantization
def profile(model, input_size, quantization_rate, custom_ops={}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0:
            return
        m.register_buffer("total_ops", torch.zeros(1))
        m.register_buffer("total_params", torch.zeros(1))
        m.register_buffer("storage_params", torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()])

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(partial(count_conv2d, r=quantization_rate))
            for p in m.parameters():
                m.storage_params += (
                    torch.Tensor([p.numel()]) * quantization_rate
                )
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
            for p in m.parameters():
                m.storage_params += torch.Tensor([p.numel()]) / 2
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
            for p in m.parameters():
                m.storage_params += torch.Tensor([p.numel()]) / 2
        elif isinstance(m, (nn.AvgPool2d)):
            m.register_forward_hook(count_avgpool)
            for p in m.parameters():
                m.storage_params += torch.Tensor([p.numel()]) / 2
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(partial(count_linear, r=quantization_rate))
            for p in m.parameters():
                m.storage_params += (
                    torch.Tensor([p.numel()]) * quantization_rate
                )
        elif isinstance(m, nn.Sequential):
            m.register_forward_hook(count_sequential)
            for p in m.parameters():
                m.storage_params += torch.Tensor([p.numel()]) / 2
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    storage_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0:
            continue
        total_ops += m.total_ops
        total_params += m.total_params
        storage_params += m.storage_params

    return (
        int(total_ops.item()),
        int(total_params.item()),
        int(storage_params.item()),
    )


def get_micronet_score(sota_model, sota_bits):
    ref_model = PreActResNet()

    ref_bits = 16

    quantization_rate = sota_bits / ref_bits
    ref_ops, ref_params, ref_storage = profile(ref_model, (1, 3, 32, 32), 0.5)
    sota_ops, sota_params, sota_storage = profile(
        sota_model, (1, 3, 32, 32), quantization_rate
    )
    print(
        f"[Ref] n_ops: {ref_ops:,} / n_params: {ref_params:,} / ",
        f"ref_storage: {ref_storage:,} / ref_bits: {ref_bits}",
    )
    print(
        f"[SOTA] n_ops: {sota_ops:,} / n_params: {sota_params:,} / ",
        f"sota_storage: {sota_storage:,} / sota_bits: {sota_bits}\n",
    )

    score_ops = sota_ops / ref_ops
    score_params = sota_storage / ref_storage
    score = score_ops + score_params
    print(
        f"Global score: {score:.2E} ",
        f"(score_ops: {score_ops:.2E} / score_params: {score_params:.2E}",
        f"quantization_rate: {quantization_rate:.2f})",
    )
