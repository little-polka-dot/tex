import torch
import torch.nn as nn
import torch.optim as optim
import tex.models.structure.losses as losses
from torch.utils.data import DataLoader
import tex.utils.builder as builder
from torch.utils.data import Dataset
import numpy as np


def train_structure(model: nn.Module, dataloader: DataLoader,
                    device_ids=None, lr=0.0001, epochs=10):
    model = model.to(torch.double)
    if device_ids:
        if isinstance(device_ids, list):
            model = nn.DataParallel(
                model, device_ids=device_ids)
        else:
            model = model.to(device_ids)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        for (x_data, seq_inputs), (seq_labels, seq_pos) in dataloader:
            optimizer.zero_grad()
            cls_x, box_x = model(x_data, seq_inputs, False)
            cls_loss, iou_loss = \
                losses.structure_loss((cls_x, box_x), (seq_labels, seq_pos))
            print(cls_loss, iou_loss)
            (cls_loss + iou_loss).backward()
            optimizer.step()
        # model.eval()
        # with torch.no_grad():
        #     pass
    torch.save(model, './test.pt')


class RandomDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        # x_data [bs, enc_len, 4] y_data description:[bs, dec_len] position:[bs, dec_len, 4]

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        x_data = np.random.random((20, 4)) * 100
        y_data = {
            'description': {
                'rows': 2,
                'cols': 10,
                'data': [['CELL'] * 10] * 2
            },
            'position': x_data
        }
        if callable(self.transform):
            x_data, y_data = self.transform(x_data, y_data)
        return x_data, y_data


def test():
    settings = {
        'model': {
            'class': 'tex.models.structure.PosStructure',
            'd_input': 42,
            'd_model': 32,
            'enc_layers': 4,
            'n_vocab': 9,
            'dec_len': 11,
            'n_head': 8,
            'd_k': 32,
            'd_ffn': 32,
            'dec_n_pos': 512,
            'dec_layers': 4,
            'pad_idx': 0,
            'dropout': 0.1
        },
        'dataloader': {
            'class': 'torch.utils.data.DataLoader',
            'dataset': {
                'class': '.RandomDataset',
                'transform': {
                    'class': 'tex.datasets.transform.PositionalStructureTransform',
                    'enc_len': 50,
                    'dec_len': 50,
                    'normalize_position': True,
                    'transform_position': True
                }
            },
            'batch_size': 10,
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
