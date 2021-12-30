from tex.datasets import structure
from torch.utils.data import DataLoader


def build_simple_structure(cfg):
    return DataLoader(
        structure.SimpleStructureDataset(**cfg['dataset']),
        batch_size=cfg['batch_size'],
        shuffle=cfg['shuffle'],  # 每次epoch前打乱数据顺序
        num_workers=cfg['num_workers'],  # 读取数据的进程数
        drop_last=cfg['drop_last'],  # 过滤数据量小于batch_size的批次
        timeout=cfg['timeout'],  # 等待worker进程的超时时间
    )


if __name__ == '__main__':
    loader = build_simple_structure({
        'dataset': {
            'path': 'E:/Code/Mine/github/tex/test/.data/structure/train',
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
    })
    for _ in range(3):
        for x, y in loader:
            print(x[0].size())
            print(x[1].size())
            print(y[0].size())
            print(y[1].size())
