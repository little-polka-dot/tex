from tex.datasets import structure
from torch.utils.data import DataLoader
from tex.datasets import transform


def build_structure_dataset(cls, *args,  **kwargs):
    return getattr(structure, cls)(*args, **kwargs)


def build_structure_transform(cls, *args, **kwargs):
    return getattr(transform, cls)(*args, **kwargs)


def build_dataloader(*args, **kwargs):
    return DataLoader(*args, **kwargs)


if __name__ == '__main__':
    loader = build_dataloader(
        build_structure_dataset(
            'SimpleStructureDataset',
            path='E:/Code/Mine/github/tex/test/.data/structure/train',
            transform=build_structure_transform(
                'StructureTransformer',
                seq_len=256,
                image_size=224,
                normalize_position=True,
                gaussian_noise=False,
                flim_mode=False,
                gaussian_blur=[
                    {'kernel': 3, 'sigma': 0},
                ],
                threshold=None,
            )
        ),
        batch_size=5,
        shuffle=True,
        num_workers=3,
        drop_last=True,
        timeout=0,
    )
    for _ in range(3):
        for x, y in loader:
            print(x[0].size())
            print(x[1].size())
            print(y[0].size())
            print(y[1].size())
