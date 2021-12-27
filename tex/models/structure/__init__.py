import torch.nn as nn
from tex.models.structure.encoder import Encoder
from tex.models.structure.decoder import Decoder
from tex.models.structure.decoder.attention import PositionalEncoding
from tex.models.structure.encoder import residual


class TexStructure(nn.Module):

    def __init__(self, **kwargs):

        super(TexStructure, self).__init__()

        self.enc_net = nn.Sequential(
            Encoder(
                d_input=kwargs['encoder'].get('d_input'),
                d_model=kwargs['encoder'].get('d_model'),
                block=kwargs['encoder'].get('block'),
                layers=kwargs['encoder'].get('layers')
            ),
            PositionalEncoding(
                kwargs['encoder'].get('d_model'),
                kwargs['encoder'].get('n_position')
            ),
        )

        self.dec_net = Decoder(
            n_vocab=kwargs['decoder'].get('n_vocab'),
            seq_len=kwargs['decoder'].get('seq_len'),
            d_model=kwargs['decoder'].get('d_model'),
            n_head=kwargs['decoder'].get('n_head'),
            d_k=kwargs['decoder'].get('d_k'),
            h_layers=kwargs['decoder'].get('h_layers'),
            d_layers=kwargs['decoder'].get('d_layers'),
            pad_idx=kwargs['decoder'].get('pad_idx'),
            d_ffn=kwargs['decoder'].get('d_ffn'),
            dropout=kwargs['decoder'].get('dropout'),
            n_position=kwargs['decoder'].get('n_position')
        )

    def forward(self, enc_input, dec_input):
        return self.dec_net(dec_input, self.enc_net(enc_input))
