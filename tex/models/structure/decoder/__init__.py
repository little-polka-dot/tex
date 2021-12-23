
import numpy as np
import torch
import torch.nn as nn
from tex.models.structure.decoder import attention


def pad_mask(x, pad_idx=0):
    # [batch_size, sql_len] -> [batch_size, 1, sql_len]
    return (x != pad_idx).unsqueeze(-2)


def sub_mask(x, pad_idx=0):
    # [batch_size, sql_len] -> [batch_size, sql_len, sql_len]
    return pad_mask(x, pad_idx) & (torch.tril(torch.ones(
        (1, x.size(1), x.size(1)), device=x.device))).bool()


def sos(batch_size, sos_idx):  # [batch_size, 1]
    """
    初始化decoder输入数据
    """
    return torch.tensor([sos_idx] * batch_size).unsqueeze(-1)


class Decoder(nn.Module):

    def __init__(self, n_vocab, seq_len, d_model, n_head, d_k,
                 h_layers, d_layers, pad_idx=0, dropout=0.1, n_position=256):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, d_model, padding_idx=pad_idx)
        self.pos_enc = attention.PositionalEncoding(d_model, n_position)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.h_decoders = nn.ModuleList([
            attention.DecodeLayer(
                d_model, n_head, d_k, dropout=dropout) for _ in range(h_layers)
        ])
        self.c_decoders = nn.ModuleList([
            attention.DecodeLayer(
                d_model, n_head, d_k, dropout=dropout) for _ in range(d_layers)
        ])
        self.b_decoders = nn.ModuleList([
            attention.DecodeLayer(
                d_model, n_head, d_k, dropout=dropout) for _ in range(d_layers)
        ])
        self.cls_fc = nn.Linear(d_model, n_vocab, bias=False)
        self.box_fc = nn.Linear(d_model, 4)
        self.pad_idx, self.d_model, self.seq_len = pad_idx, d_model, seq_len

    def dec_mask(self, dec_input):
        return sub_mask(dec_input, self.pad_idx)

    def dec_embedding(self, dec_input):
        emb_value = self.embedding(dec_input) * (self.d_model ** 0.5)
        return self.norm(self.dropout(self.pos_enc(emb_value)))

    def decode(self, dec_input, enc_value):
        slf_m = self.dec_mask(dec_input)
        out_x = self.dec_embedding(dec_input)
        for layer in self.h_decoders:
            out_x = layer(out_x, enc_value, slf_mask=slf_m)
        cls_x, box_x = out_x, out_x
        for layer in self.c_decoders:
            cls_x = layer(cls_x, enc_value, slf_mask=slf_m)
        cls_x = self.cls_fc(cls_x)
        for layer in self.b_decoders:
            box_x = layer(box_x, enc_value, slf_mask=slf_m)
        box_x = self.box_fc(box_x)
        return cls_x, torch.sigmoid(box_x)

    def greedy_decode(self, dec_input, enc_value):
        cls_x, box_x = None, None
        for _ in range(self.seq_len):
            cls_x, box_x = self.decode(dec_input, enc_value)
            _, next_word = torch.max(cls_x, dim=-1)
            dec_input = torch.cat([
                dec_input, next_word[:, -1].unsqueeze(-1)], dim=-1)
        # cls_x: [batch_size, seq_len, n_vocab]
        # box_x: [batch_size, seq_len, 4]
        return cls_x, box_x

    def forward(self, dec_input, enc_value):
        # dec_input: [batch_size, dec_len]
        # enc_value: [batch_size, enc_len, d_model]
        if self.training:
            return self.decode(dec_input, enc_value)
        else:
            return self.greedy_decode(dec_input, enc_value)


if __name__ == '__main__':

    # https://www.jianshu.com/p/f8955dbc3553 训练过程

    def test():
        batch_size = 2
        seq_len = 10
        n_vocab = 8
        d_model = 17
        n_head = 4
        d_k = 16
        decoder = Decoder(n_vocab, seq_len, d_model, n_head, d_k, 3, 1)
        enc_v = torch.tensor(np.random.random([batch_size, 21, d_model]), dtype=torch.float)
        print(sos(batch_size, 1).size())
        decoder.eval()
        cls_x, box_x = decoder(sos(batch_size, 1), enc_v)
        _, next_word = torch.max(cls_x, dim=-1)
        print(next_word)
        print(cls_x)
        # 输入特征图 [batch_size, 单层特征图一维, 特征图深度]

    test()
