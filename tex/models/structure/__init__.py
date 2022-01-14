import torch.nn as nn
from tex.models.structure.encoder import BackboneEncoder
from tex.models.structure.decoder import Decoder
from tex.models.structure.attention import PositionalEncoding


class BackboneStructure(nn.Module):

    def __init__(self, im_channels, d_model, enc_block, enc_layers,
                 n_vocab, seq_len, n_head, d_k, d_ffn, enc_n_pos, dec_n_pos,
                 dec_layers=3, dec_sp_layers=1, pad_idx=0, dec_dropout=0.1):

        super(BackboneStructure, self).__init__()

        self.enc_net = nn.Sequential(
            BackboneEncoder(
                d_input=im_channels,
                d_model=d_model,
                block=enc_block,
                layers=enc_layers
            ),
            PositionalEncoding(d_model, enc_n_pos),
        )

        self.dec_net = Decoder(
            n_vocab=n_vocab,
            seq_len=seq_len,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            layers=dec_layers,
            sp_layers=dec_sp_layers,
            pad_idx=pad_idx,
            d_ffn=d_ffn,
            dropout=dec_dropout,
            n_position=dec_n_pos
        )

    def forward(self, enc_input, dec_input):
        return self.dec_net(dec_input, self.enc_net(enc_input))
