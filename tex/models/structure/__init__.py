import torch.nn as nn


class BackboneStructure(nn.Module):

    def __init__(self, im_channels, d_model, enc_block, enc_layers,
                 n_vocab, dec_len, n_head, d_k, d_ffn, enc_n_pos, dec_n_pos,
                 dec_layers, dec_sp_layers=1, pad_idx=0, dropout=0.1):

        super(BackboneStructure, self).__init__()

        from tex.models.structure.encoder import BackboneEncoder as Encoder
        from tex.models.structure.decoder import Decoder
        from tex.models.transformer.attention import PositionalEncoding

        self.enc_net = nn.Sequential(
            Encoder(
                d_input=im_channels,
                d_model=d_model,
                block=enc_block,
                layers=enc_layers
            ),
            PositionalEncoding(d_model, enc_n_pos),
        )

        self.dec_net = Decoder(
            n_vocab=n_vocab,
            seq_len=dec_len,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            layers=dec_layers,
            sp_layers=dec_sp_layers,
            pad_idx=pad_idx,
            d_ffn=d_ffn,
            dropout=dropout,
            n_position=dec_n_pos
        )

    def forward(self, enc_input, dec_input):
        return self.dec_net(dec_input, self.enc_net(enc_input))


class TransformerStructure(nn.Module):

    def __init__(self, d_input, d_model, enc_layers, n_vocab, dec_len,
                 n_head, d_k, d_ffn, dec_n_pos,
                 dec_layers, dec_sp_layers=1, pad_idx=0, dropout=0.1):

        super(TransformerStructure, self).__init__()

        from tex.models.structure.encoder import TransformerEncoder as Encoder
        from tex.models.structure.decoder import Decoder

        self.enc_net = Encoder(
            d_input=d_input,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            layers=enc_layers,
            dropout=dropout,
            d_ffn=d_ffn
        )

        self.dec_net = Decoder(
            n_vocab=n_vocab,
            seq_len=dec_len,
            d_model=d_model,
            n_head=n_head,
            d_k=d_k,
            layers=dec_layers,
            sp_layers=dec_sp_layers,
            pad_idx=pad_idx,
            d_ffn=d_ffn,
            dropout=dropout,
            n_position=dec_n_pos
        )

    def forward(self, enc_input, dec_input):
        enc_value, mask = self.enc_net(enc_input)
        return self.dec_net(
            dec_input, enc_value, mask=mask)

