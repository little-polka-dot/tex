import torch
import torch.nn as nn
from tex.models.structure import attention


class Decoder(nn.Module):

    def __init__(self, n_vocab, seq_len, d_model, n_head, d_k, layers, sp_layers,
                 pad_idx=0, dropout=0.1, d_ffn=None, n_position=256):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, d_model, padding_idx=pad_idx)
        self.pos_enc = attention.PositionalEncoding(d_model, n_position)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.h_decoders = nn.ModuleList([
            attention.DecodeLayer(
                d_model, n_head, d_k, d_ffn=d_ffn, dropout=dropout) for _ in range(layers)
        ])
        self.c_decoders = nn.ModuleList([
            attention.DecodeLayer(
                d_model, n_head, d_k, d_ffn=d_ffn, dropout=dropout) for _ in range(sp_layers)
        ])
        self.b_decoders = nn.ModuleList([
            attention.DecodeLayer(
                d_model, n_head, d_k, d_ffn=d_ffn, dropout=dropout) for _ in range(sp_layers)
        ])
        self.cls_fc = nn.Linear(d_model, n_vocab, bias=False)
        self.box_fc = nn.Linear(d_model, 4)
        self.pad_idx, self.d_model, self.seq_len = pad_idx, d_model, seq_len

    def decode(self, dec_input, enc_value, mask=None):
        slf_m = attention.sub_mask(dec_input, self.pad_idx)
        out_x = self.norm(
            self.dropout(
                self.pos_enc(
                    self.embedding(dec_input) * (self.d_model ** 0.5)
                )
            )
        )
        for layer in self.h_decoders:
            out_x = layer(out_x, enc_value, slf_mask=slf_m, enc_mask=mask)
        cls_x, box_x = out_x, out_x
        for layer in self.c_decoders:
            cls_x = layer(cls_x, enc_value, slf_mask=slf_m, enc_mask=mask)
        cls_x = self.cls_fc(cls_x)
        for layer in self.b_decoders:
            box_x = layer(box_x, enc_value, slf_mask=slf_m, enc_mask=mask)
        box_x = self.box_fc(box_x)
        return cls_x, torch.sigmoid(box_x)

    def greedy_decode(self, dec_input, enc_value, mask=None):
        cls_x, box_x = None, None
        for _ in range(self.seq_len):
            cls_x, box_x = self.decode(
                dec_input, enc_value, mask=mask)
            next_word = torch.argmax(cls_x, dim=-1)[:, -1]
            dec_input = torch.cat([
                dec_input, next_word.unsqueeze(-1)], dim=-1)
        # cls_x: [batch_size, seq_len, n_vocab]
        # box_x: [batch_size, seq_len, 4]
        return cls_x, box_x

    def forward(self, dec_input, enc_value, mask=None):
        # dec_input: [batch_size, dec_len]
        # enc_value: [batch_size, enc_len, d_model]
        if self.training:
            return self.decode(
                dec_input, enc_value, mask=mask)
        else:
            return self.greedy_decode(
                dec_input, enc_value, mask=mask)


# if __name__ == '__main__':
#     x = torch.randn((2, 6, 3))
#     print(x)
#     print('1', torch.max(x, dim=-1)[1])
#     print('2', torch.argmax(x, dim=-1))
#
#     print(torch.argmax(x, dim=-1).eq(torch.max(x, dim=-1)[1]).sum()/12)
