import torch
import torch.nn as nn
import torch.optim as optim
import tex.models.structure.losses as losses
from torch.utils.data import DataLoader
import tex.utils.builder as builder
from torch.utils.data import Dataset
import random
import numpy as np


def train_structure(model: nn.Module, dataloader: DataLoader,
                    device_ids=None, lr=0.0001, epochs=10):
    if device_ids:
        if isinstance(device_ids, list):
            model = nn.DataParallel(
                model, device_ids=device_ids)
        else:
            model = model.to(device_ids)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.double()
    for epoch in range(epochs):
        model.train()
        for input_, target in dataloader:
            optimizer.zero_grad()
            output = model(*input_)
            cls_loss, iou_loss = \
                losses.structure_loss(output, target)
            print(cls_loss, iou_loss)
            p = [i for i in model.enc_net.pos_feed.parameters()]
            (cls_loss + iou_loss).backward()
            optimizer.step()
            p = [i for i in model.enc_net.pos_feed.parameters()]
        # model.eval()
        # with torch.no_grad():
        #     pass
    torch.save(model, './test.pt')


class RandomDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # x_data [bs, enc_len, 4] y_data description:[bs, dec_len] position:[bs, dec_len, 4]

    def __len__(self):
        return 5

    def __getitem__(self, index):
        x_data = np.array([
            [0, 0, 1, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1]
        ])
        y_data = {
            'description': {
                'rows': 2,
                'cols': 2,
                'data': [['CELL'] * 2] * 2
            },
            'position': x_data
        }
        if callable(self.transform):
            x_data, y_data = self.transform(x_data, y_data)
        return x_data, y_data


def test():
    settings = {
        'model': {
            'class': 'tex.models.structure.PositionalStructure',
            'd_input': 4,
            'd_model': 16,
            'enc_layers': 4,
            'n_vocab': 9,
            'dec_len': 11,
            'n_head': 8,
            'd_k': 32,
            'd_ffn': 32,
            'dec_n_pos': 512,
            'dec_layers': 3,
            'dec_tail_layers': 1,
            'pad_idx': 0,
            'dropout': 0.1
        },
        'dataloader': {
            'class': 'torch.utils.data.DataLoader',
            'dataset': {
                'class': '.RandomDataset',
                'transform': {
                    'class': 'tex.datasets.transform.PositionalStructureTransform',
                    'enc_len': 10,
                    'dec_len': 10,
                    'normalize_position': True,
                    'transform_position': False
                }
            },
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 0,
            'drop_last': True,
            'timeout': 0,
        },
        'device_ids': None,
        'lr': 0.0001,
        'epochs': 100
    }
    train_structure(**builder.build_from_settings(settings))


if __name__ == '__main__':
    test()
