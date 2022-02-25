import torch.nn as nn


class ResStructure(nn.Module):

    def __init__(self, d_input, d_model, n_vocab,
                 dec_len, n_head, d_k, d_ffn, enc_n_pos, dec_n_pos,
                 dec_layers, pad_idx=0, dropout=0.1):

        super(ResStructure, self).__init__()

        from tex.models.structure.encoder import ResEncoder as Encoder
        from tex.models.structure.decoder import Decoder
        from tex.models.transformer.attention import PositionalEncoding

        self.enc_net = nn.Sequential(
            Encoder(d_input=d_input, layers=(1, 2, 5, 3)),
            PositionalEncoding(d_model, enc_n_pos),
        )

        self.dec_net = Decoder(
            n_vocab=n_vocab,
            seq_len=dec_len,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_ffn=d_ffn,
            layers=dec_layers,
            pad_idx=pad_idx,
            dropout=dropout,
            n_position=dec_n_pos
        )

    def forward(self, enc_input, dec_input, is_greedy=False):
        return self.dec_net(
            dec_input, self.enc_net(enc_input), None, is_greedy)


class PosStructure(nn.Module):

    def __init__(self, d_input, d_model, enc_layers, n_vocab,
                 dec_len, n_head, d_k, d_ffn, dec_n_pos,
                 dec_layers, pad_idx=0, dropout=0.1):

        super(PosStructure, self).__init__()

        from tex.models.structure.encoder import PosEncoder as Encoder
        from tex.models.structure.decoder import Decoder

        self.enc_net = Encoder(
            d_input=d_input,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_ffn=d_ffn,
            layers=enc_layers,
            dropout=dropout,
        )

        self.dec_net = Decoder(
            n_vocab=n_vocab,
            seq_len=dec_len,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            d_ffn=d_ffn,
            layers=dec_layers,
            pad_idx=pad_idx,
            dropout=dropout,
            n_position=dec_n_pos
        )

    def forward(self, enc_input, dec_input, is_greedy=False):
        enc_value, enc_mask = self.enc_net(enc_input)
        return self.dec_net(
            dec_input, enc_value, enc_mask, is_greedy)

