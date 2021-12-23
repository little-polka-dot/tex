import numpy as np
import torch
import torch.nn as nn
from tex.models.structure import encoder
from tex.models.structure import decoder
from tex.models.structure.encoder import resnet


class Tex(nn.Module):

    default_kwargs = {
        'i_channels': 1,  # encoder输入序列长度
        'o_channels': 512,  # encoder输出序列长度
        'conv_layers': [3, 4, 6, 3],
        'block': resnet.Bottleneck,
        'n_vocab': 9,
        'seq_len': 128,  # decoder序列长度
        'd_feature': 1029,  # 对应图像size: 224 x 224
        'd_model': 256,  # encoder与decoder维度（须保持同步）
        'n_head': 8,
        'd_k': 128,
        'h_layers': 3,
        'd_layers': 1,
        'pad_idx': 0,
        'dropout': 0.1
    }

    def __init__(self, **kwargs):

        super(Tex, self).__init__()

        def kw_get(key):
            return kwargs.get(key, self.default_kwargs[key])

        self.enc_net = nn.Sequential(
            encoder.Encoder(
                i_channels=kw_get('i_channels'),
                o_channels=kw_get('o_channels'),
                block=kw_get('block'),
                conv_layers=kw_get('conv_layers')
            ),
            nn.Linear(
                kw_get('d_feature'), kw_get('d_model')),
            decoder.attention.PositionalEncoding(
                kw_get('d_model'), kw_get('o_channels'))
        )

        self.dec_net = decoder.Decoder(
            n_vocab=kw_get('n_vocab'),
            seq_len=kw_get('seq_len'),
            d_model=kw_get('d_model'),
            n_head=kw_get('n_head'),
            d_k=kw_get('d_k'),
            h_layers=kw_get('h_layers'),
            d_layers=kw_get('d_layers'),
            pad_idx=kw_get('pad_idx'),
            dropout=kw_get('dropout'),
        )

    def forward(self, enc_input, dec_input):
        return self.dec_net(dec_input, self.enc_net(enc_input))


if __name__ == '__main__':
    batch_size = 1
    sos_idx = 1
    pad_idx = 0
    enc_x = torch.tensor(np.random.random((batch_size, 1, 224, 224)), dtype=torch.float)
    dec_x = decoder.sos(batch_size, sos_idx)
    net = Tex(pad_idx=pad_idx)
    net.eval()
    cls_x, box_x = net(enc_x, dec_x)
    print(cls_x.size())
    print(box_x.size())

