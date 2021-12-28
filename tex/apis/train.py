import torch
import torch.nn as nn
import torch.optim as optim
from tex.models import builder


def train_structure(cfg, device_ids=None):

    net = builder.build_structure(cfg)
    if device_ids:
        if len(device_ids) > 1:
            net = nn.DataParallel(net, device_ids=device_ids)
        else:
            net = net.to(device_ids[0])


if __name__ == '__main__':
    structure_cfg = {
        'encoder': {
            'd_input': 1,  # encoder输入图层数
            'd_model': 512,  # encoder输出图层数
            'layers': [3, 4, 6, 3],
            'block': 'ContextualBlock',
            'n_position': 4096,  # 须大于等于图像卷积后的size
        },
        'decoder': {
            'd_model': 512,  # decoder维度
            'n_vocab': 9,  # 表结构描述语言词汇量
            'seq_len': 256,  # decoder序列长度
            'n_head': 8,
            'd_k': 128,
            'h_layers': 5,
            'd_layers': 1,
            'd_ffn': 1024,
            'dropout': 0.1,
            'n_position': 256,  # 须大于等于seq_len
            'pad_idx': 0,
        }
    }
    train_structure(structure_cfg)
