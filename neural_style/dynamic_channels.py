import random
import math
import torch
from ops import *
from transformer_net import TransformerNet,ResidualBlock, T_CHANNEL_CONFIG
CHANNEL_CONFIGS = [0.25, 0.5, 0.75, 1.0]


def get_full_channel_configs(model):
    full_channels = []
    for m in model.modules():
        if isinstance(m, ConstantInput):
            full_channels.append(m.input.shape[1])
        elif isinstance(m, EqualConv2d):
            #if m.weight.shape[1] == 3 and m.weight.shape[-1] == 1:
                #continue
            full_channels.append(m.weight.shape[0])  # get the output channels
        elif isinstance(m, Instancenorm):
            full_channels.append(m.gamma.shape[1])
    return full_channels


def set_sub_channel_config(model, sub_channels):
    ptr = 0
    #model_modules = [x for x in  model.modules()]
    #print(model_modules)
    for m in model.modules():
        if isinstance(m, EqualConv2d):
            #print('conv',sub_channels[ptr])
            #if m.weight.shape[1] == 3 and m.weight.shape[-1] == 1:
                #continue
            m.first_k_oup = max(sub_channels[ptr],3)
            ptr += 1
        elif isinstance(m, Instancenorm):
            #print('instance',sub_channels[ptr])
            m.first_k_oup = sub_channels[ptr]
            ptr += 1
    assert ptr == len(sub_channels), (ptr, len(sub_channels))  # all used


def set_uniform_channel_ratio(model, ratio):
    full_channels = get_full_channel_configs(model)

    channel_config = [min(v, int(v * ratio)) for v in T_CHANNEL_CONFIG]

    set_sub_channel_config(model, channel_config)


def remove_sub_channel_config(model):
    for m in model.modules():
        if hasattr(m, 'first_k_oup'):
            del m.first_k_oup


def reset_generator(model):
    remove_sub_channel_config(model)
    if hasattr(model, 'target_res'):
        del model.target_res


def get_current_channel_config(model):
    ch = []
    for m in model.modules():
        if hasattr(m, 'first_k_oup'):
            ch.append(m.first_k_oup)
    return ch


def _get_offical_sub_channel_config(ratio, org_channel_mult):
    channel_max = 512
    # NOTE: in Python 3.6 onwards, the order of dictionary insertion is preserved
    channel_config = [min(channel_max, int(v * ratio * org_channel_mult)) for _, v in T_CHANNEL_CONFIG.items()]
    channel_config2 = []  # duplicate the config
    for c in channel_config:
        channel_config2.append(c)
        channel_config2.append(c)
    return channel_config2


def get_random_channel_config(full_channels, org_channel_mult, min_channel=8, divided_by=1):
    # use the official config as the smallest number here (so that we can better compare the computation)
    bottom_line = _get_offical_sub_channel_config(CHANNEL_CONFIGS[0], org_channel_mult)
    bottom_line = bottom_line[:len(full_channels)]

    new_channels = []
    ratios = []
    for full_c, bottom in zip(full_channels, bottom_line):
        valid_channel_configs = [a for a in CHANNEL_CONFIGS if a * full_c >= bottom]  # if too small, discard the ratio
        ratio = random.choice(valid_channel_configs)
        ratios.append(ratio)
        c = int(ratio * full_c)
        c = min(max(c, min_channel), full_c)
        c = math.ceil(c * 1. / divided_by) * divided_by
        new_channels.append(c)
    return new_channels, ratios


def sample_random_sub_channel(model, min_channel=8, divided_by=1, seed=None, mode='uniform', set_channels=True):
    if seed is not None:  # whether to sync between workers
        random.seed(seed)

    if mode == 'uniform':
        # case 1: sample a uniform channel config
        rand_ratio = random.choice(CHANNEL_CONFIGS)
        # print(rand_ratio)
        if set_channels:
            set_uniform_channel_ratio(model, rand_ratio)
        return [rand_ratio] * len(get_full_channel_configs(model))
    elif mode == 'flexible':
        # case 2: sample flexible per-channel ratio
        full_channels = get_full_channel_configs(model)
        org_channel_mult = full_channels[-1] / T_CHANNEL_CONFIG[model.resolution]
        rand_channels, rand_ratios = get_random_channel_config(full_channels, org_channel_mult, min_channel, divided_by)
        if set_channels:
            set_sub_channel_config(model, rand_channels)
        return rand_ratios
    elif mode == 'sandwich':
        # case 3: sandwich sampling for flexible ratio setting
        rrr = random.random()
        if rrr < 0.25:  # largest
            if set_channels:
                remove_sub_channel_config(model)  # i.e., use the largest channel
            return [CHANNEL_CONFIGS[-1]] * len(get_full_channel_configs(model))
        elif rrr < 0.5:  # smallest
            if set_channels:
                set_uniform_channel_ratio(model, CHANNEL_CONFIGS[0])
            return [CHANNEL_CONFIGS[0]] * len(get_full_channel_configs(model))
        else:  # random sample
            full_channels = get_full_channel_configs(model)
            org_channel_mult = full_channels[-1] / T_CHANNEL_CONFIG[model.resolution]
            rand_channels, rand_ratios = get_random_channel_config(full_channels, org_channel_mult, min_channel,
                                                                   divided_by)
            if set_channels:
                set_sub_channel_config(model, rand_channels)
            return rand_ratios
    else:
        raise NotImplementedError


def sort_channel(g):
    assert isinstance(g, TransformerNet)
    def _get_sorted_input_idx(conv):
        assert isinstance(conv, EqualConv2d)
        importance = torch.sum(torch.abs(conv.weight.data), dim=(0, 1, 3, 4))
        return torch.sort(importance, dim=0, descending=True)[1]

    def _reorg_input_channel(conv, idx):
        assert idx.numel() ==conv.weight.data.shape[2]
        conv.weight.data = torch.index_select(conv.weight.data, 2, idx)  # inp
        conv.modulation.weight.data = torch.index_select(conv.weight.data, 0, idx)
        conv.bias.data = conv.bias.data[idx]

    def _reorg_output_channel(conv, idx):
        assert idx.numel() == conv.weight.data.shape[1]
        conv.weight.data = torch.index_select(conv.conv.weight.data, 1, idx)  # oup
        conv.bias.data = conv.bias.data[idx]

    # NOTE:
    # 1. MLP does not need to be changed
    # 2. Instancenorm ?
    sorted_idx = None # get the input latents
    sorted_idx = _get_sorted_input_idx(g.deconv3.conv2d)
    _reorg_input_channel(g.deconv3.conv2d, sorted_idx)
    _reorg_output_channel(g.deconv2.conv2d, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.deconv2.conv2d)
    _reorg_input_channel(g.deconv2.conv2d, sorted_idx)
    _reorg_output_channel(g.deconv1.conv2d, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.deconv1.conv2d)
    _reorg_input_channel(g.deconv1.conv2d, sorted_idx)

    _reorg_output_channel(g.res5.conv2, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res5.conv2.conv2d)
    _reorg_input_channel(g.res5.conv2, sorted_idx)
    _reorg_output_channel(g.res5.conv1, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res5.conv1.conv2d)
    _reorg_input_channel(g.res5.conv1, sorted_idx)
    _reorg_output_channel(g.res4.conv2, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res4.conv2.conv2d)
    _reorg_input_channel(g.res4.conv2, sorted_idx)
    _reorg_output_channel(g.res4.conv1, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res4.conv1)
    _reorg_input_channel(g.res4.conv1, sorted_idx)
    _reorg_output_channel(g.res3.conv2, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res3.conv2.conv2d)
    _reorg_input_channel(g.res3.conv2, sorted_idx)
    _reorg_output_channel(g.res3.conv1, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res3.conv1)
    _reorg_input_channel(g.res3.conv1, sorted_idx)
    _reorg_output_channel(g.res2.conv2, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res2.conv2.conv2d)
    _reorg_input_channel(g.res2.conv2, sorted_idx)
    _reorg_output_channel(g.res2.conv1, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res2.conv1)
    _reorg_input_channel(g.res2.conv1, sorted_idx)
    _reorg_output_channel(g.res1.conv2, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res1.conv2.conv2d)
    _reorg_input_channel(g.res1.conv2, sorted_idx)
    _reorg_output_channel(g.res1.conv1, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.res1.conv1)
    _reorg_input_channel(g.res1.conv1, sorted_idx)

    sorted_idx = _get_sorted_input_idx(g.conv3.conv2d)
    _reorg_input_channel(g.conv3.conv2d, sorted_idx)
    _reorg_output_channel(g.conv2.conv2d, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.conv2.conv2d)
    _reorg_input_channel(g.conv2.conv2d, sorted_idx)
    _reorg_output_channel(g.conv1.conv2d, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.conv1.conv2d)

    # sort conv1
    _reorg_output_channel(g.conv1, sorted_idx)
    sorted_idx = _get_sorted_input_idx(g.conv1.conv2d)
    _reorg_input_channel(g.conv1, sorted_idx)



