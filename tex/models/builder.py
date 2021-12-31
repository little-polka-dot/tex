from tex.models import structure


def build_structure(cls, *args, **kwargs):
    return getattr(structure, cls)(*args, **kwargs)


if __name__ == '__main__':
    import torch
    from tex.models.structure.decoder import sos
    batch_size = 1
    sos_idx = 1
    pad_idx = 0
    enc_x = torch.randn((batch_size, 1, 224, 224), dtype=torch.float)
    dec_x = sos(batch_size, sos_idx)
    net = build_structure(
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
    net.eval()
    cls_x, box_x = net(enc_x, dec_x)
    print(cls_x.size())
    print(box_x.size())
