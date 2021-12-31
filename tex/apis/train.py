import torch
import torch.nn as nn
import torch.optim as optim
import tex.models as models
import tex.datasets as datasets


def train_structure(model, dataloader, device_ids=None, lr=0.0001, epochs=10):
    from tex.models.structure.losses import structure_loss
    if device_ids:
        if isinstance(device_ids, list):
            model = nn.DataParallel(
                model, device_ids=device_ids)
        else:
            model = model.to(device_ids)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        model.train()
        for input_, target in dataloader:
            optimizer.zero_grad()
            output = model(*input_)
            loss = structure_loss(output, target)
            loss.backward()
            optimizer.step()
        # model.eval()
        # with torch.no_grad():
        #     pass


def test():
    model = models.builder.build_structure(
        'TexStructure',
        im_channels=1,  # encoder输入图层数
        d_model=512,  # encoder输出图层数
        enc_layers=[3, 4, 6, 3],
        enc_block='ContextualBlock',
        enc_n_pos=4096,  # 须大于等于图像卷积后的size
        n_vocab=9,  # 表结构描述语言词汇量
        seq_len=256,  # decoder序列长度
        n_head=8,
        d_k=128,
        dec_layers=5,
        dec_sp_layers=1,
        d_ffn=1024,
        dec_dropout=0.1,
        dec_n_pos=256,  # 须大于等于seq_len
        pad_idx=0,
    )
    dataloader = datasets.builder.build_dataloader(
        datasets.builder.build_structure_dataset(
            'SimpleStructureDataset',
            path='E:/Code/Mine/github/tex/test/.data/structure/train',
            transform=datasets.builder.build_structure_transform(
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
    train_structure(model, dataloader)

if __name__ == '__main__':
    test()
