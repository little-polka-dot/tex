import torch.nn as nn
import torch.optim as optim
import tex.models.structure.losses as losses
from torch.utils.data import DataLoader
import tex.utils.builder as builder


def train_structure(model: nn.Module, dataloader: DataLoader,
                    device_ids=None, lr=0.0001, epochs=10):
    if device_ids:
        if isinstance(device_ids, list):
            model = nn.DataParallel(
                model, device_ids=device_ids)
        else:
            model = model.to(device_ids)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for input_, target in dataloader:
            optimizer.zero_grad()
            output = model(*input_)
            cls_loss, iou_loss = \
                losses.structure_loss(output, target)
            (cls_loss + iou_loss).backward()
            optimizer.step()
        # model.eval()
        # with torch.no_grad():
        #     pass


def test():
    settings = {
        'model': {
            'class': 'tex.models.structure.TexStructure',
            'im_channels': 1,  # encoder输入图层数
            'd_model': 512,  # 向量维度
            'enc_layers': [3, 4, 6, 3],
            'enc_block': 'ContextualBlock',
            'enc_n_pos': 4096,  # 须大于等于图像卷积后的size
            'n_vocab': 9,  # 表结构描述语言词汇量
            'seq_len': 256,  # decoder序列长度
            'n_head': 8,
            'd_k': 128,
            'dec_layers': 5,
            'dec_sp_layers': 1,
            'd_ffn': 1024,
            'dec_dropout': 0.1,
            'dec_n_pos': 256,  # 须大于等于seq_len
            'pad_idx': 0,
        },
        'dataloader': {
            'class': 'torch.utils.data.DataLoader',
            'dataset': {
                'class': 'tex.datasets.structure.SimpleStructureDataset',
                'path': 'E:/Code/Mine/github/tex/test/.data/structure/train',
                'transform': {
                    'class': 'tex.datasets.transform.StructureTransformer',
                    'seq_len': 256,
                    'image_size': 224,
                    'normalize_position': True,
                    'gaussian_noise': False,
                    'flim_mode': False,
                    'gaussian_blur': None,
                    'threshold': None,
                }
            },
            'batch_size': 5,
            'shuffle': True,
            'num_workers': 3,
            'drop_last': True,
            'timeout': 0,
        },
        'device_ids': None,
        'lr': 0.0001,
        'epochs': 10
    }
    print(builder.build_from_settings(settings))


if __name__ == '__main__':
    test()
