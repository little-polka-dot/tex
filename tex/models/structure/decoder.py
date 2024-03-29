import torch
import torch.nn as nn
import tex.models.nlp.transformer as transformer


class Decoder(nn.Module):

    def __init__(self, n_vocab, seq_len, d_model, n_head, d_k, d_ffn, layers,
                 pad_idx=0, dropout=0.1, n_position=256):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.pos_enc = transformer.PositionalEncoding(d_model, n_position)
        self.norm = nn.LayerNorm(d_model)
        self.decoders = nn.ModuleList([
            transformer.DecodeLayer(
                d_model, n_head, d_k, d_ffn, dropout=dropout) for _ in range(layers)
        ])
        # self.con_fc = nn.Linear(d_model, 1)  # 置信度
        self.cls_fc = nn.Linear(d_model, n_vocab, bias=False)
        self.box_fc = nn.Linear(d_model, 4)
        self.pad_idx, self.d_model, self.seq_len = pad_idx, d_model, seq_len

    def decode(self, dec_input, enc_value, enc_mask=None):
        slf_m = transformer.pad_mask(
            dec_input, self.pad_idx) & transformer.subsequent_mask(dec_input)
        out_x = self.norm(self.pos_enc(
            self.dropout(self.embedding(dec_input)) * (self.d_model ** 0.5)))
        for layer in self.decoders:
            out_x = layer(out_x, enc_value, slf_mask=slf_m, enc_mask=enc_mask)
        cls_x = self.cls_fc(out_x)  # 序列分类
        box_x = self.box_fc(out_x)  # 坐标回归
        return cls_x, torch.sigmoid(box_x)

    def greedy_decode(self, dec_input, enc_value, enc_mask=None):
        cls_x, box_x = None, None
        for _ in range(self.seq_len):
            cls_x, box_x = self.decode(
                dec_input, enc_value, enc_mask=enc_mask)
            next_word = torch.argmax(cls_x, dim=-1)[:, -1]
            dec_input = torch.cat([
                dec_input, next_word.unsqueeze(-1)], dim=-1)
        # cls_x: [batch_size, seq_len, n_vocab]
        # box_x: [batch_size, seq_len, 4]
        return cls_x, box_x

    def forward(self, dec_input, enc_value, enc_mask=None, is_greedy=False):
        # dec_input: [batch_size, dec_len]
        # enc_value: [batch_size, enc_len, d_model]
        if is_greedy:
            return self.greedy_decode(
                dec_input, enc_value, enc_mask=enc_mask)
        else:
            return self.decode(
                dec_input, enc_value, enc_mask=enc_mask)


# if __name__ == '__main__':
#     x = torch.randn((3,2))
#     print(x)
#     print(torch.mean(x, dim=-1))
#     print(torch.randn((3,2,1)))
#     x = torch.randn((2, 6, 3))
#     print(x)
#     print('1', torch.max(x, dim=-1)[1])
#     print('2', torch.argmax(x, dim=-1))
#
#     print(torch.argmax(x, dim=-1).eq(torch.max(x, dim=-1)[1]).sum()/12)
