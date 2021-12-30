import torch
import torch.nn as nn
import torch.optim as optim
import tex.models as models
import tex.datasets as datasets


def train_structure(cfg, device_ids=None):

    net = models.builder.build_structure(cfg['models'])
    dataloader = datasets.builder.build_simple_structure(cfg['datasets'])
    if device_ids:
        if isinstance(device_ids, list):
            net = nn.DataParallel(
                net, device_ids=device_ids)
        else:
            net = net.to(device_ids)


if __name__ == '__main__':
    config = {
        'models': {
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
        },
        'datasets': {
            'dataset': {
                'path': '.data/structure/train',
                'transform': {
                    'seq_len': 256,
                    'image_size': 227,
                    'normalize_position': True,
                    'gaussian_noise': False,
                    'flim_mode': False,
                    'gaussian_blur': [
                        {
                            'kernel': 3,
                            'sigma': 0,
                        },
                    ],
                    'threshold': None,
                }
            },
            'batch_size': 5,
            'shuffle': True,
            'num_workers': 3,
            'drop_last': True,
            'timeout': 0,
        }
    }
    train_structure(config)
