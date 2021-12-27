

def build_structure():
    from tex.models.structure import TexStructure
    return TexStructure(
        encoder={
            'd_input': 1,  # encoder输入图层数
            'd_model': 512,  # encoder输出图层数
            'layers': [3, 4, 6, 3],
            'block': 'CoTBlock',
            'n_position': 4096,  # 须大于等于图像卷积后的size
        },
        decoder={
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
    )


if __name__ == '__main__':
    import torch
    import numpy as np
    from tex.models.structure.decoder import sos
    batch_size = 1
    sos_idx = 1
    pad_idx = 0
    enc_x = torch.tensor(np.random.random((batch_size, 1, 224, 224)), dtype=torch.float)
    dec_x = sos(batch_size, sos_idx)
    net = build_structure()
    net.eval()
    cls_x, box_x = net(enc_x, dec_x)
    print(cls_x.size())
    print(box_x.size())
