import numpy as np
import torch
import torch.nn as nn
from tex.models.structure import encoder
from tex.models.structure import decoder
from tex.models.structure.encoder import resnet
from tex import settings


class TexStructure(nn.Module):

    def __init__(self, **kwargs):

        super(TexStructure, self).__init__()

        def get(key):
            return kwargs.get(key, settings.TEX_STRUCTURE_SETTINGS[key])

        self.enc_net = nn.Sequential(
            encoder.Encoder(
                d_input=get('enc_d_input'),
                d_model=get('enc_d_model'),
                block=get('enc_block'),
                layers=get('enc_layers')
            ),
            decoder.attention.PositionalEncoding(
                get('enc_d_model'), get('enc_n_position')),
        )

        self.dec_net = decoder.Decoder(
            n_vocab=get('dec_n_vocab'),
            seq_len=get('dec_seq_len'),
            d_model=get('dec_d_model'),
            n_head=get('dec_n_head'),
            d_k=get('dec_d_k'),
            h_layers=get('dec_h_layers'),
            d_layers=get('dec_d_layers'),
            pad_idx=get('pad_idx'),
            dropout=get('dec_dropout'),
            n_position=get('dec_n_position')
        )

    def forward(self, enc_input, dec_input):
        return self.dec_net(dec_input, self.enc_net(enc_input))


if __name__ == '__main__':
    batch_size = 1
    sos_idx = 1
    pad_idx = 0
    enc_x = torch.tensor(np.random.random((batch_size, 1, 224, 224)), dtype=torch.float)
    dec_x = decoder.sos(batch_size, sos_idx)
    net = TexStructure(pad_idx=pad_idx)
    net.eval()
    cls_x, box_x = net(enc_x, dec_x)
    print(cls_x.size())
    print(box_x.size())

