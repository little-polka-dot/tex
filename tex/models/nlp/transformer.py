import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    Attention(Q,K,V) = softmax(Q*K.T/sqrt(d_k))*V
    """

    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # Q,K,V: [batch_size, n_head, len_q/k/v, d_q/k/v]
        # len_k = len_v; d_q = d_k
        score = torch.matmul(
            query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        if mask is not None:
            score = score.masked_fill(mask == 0, -torch.inf)
        attn = self.dropout(torch.softmax(score, dim=-1))
        return torch.matmul(attn, value), attn


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k, self.d_v, self.n_head = d_k, d_v, n_head
        self.w_qs = nn.Linear(d_model, d_k * n_head)
        self.w_ks = nn.Linear(d_model, d_k * n_head)
        self.w_vs = nn.Linear(d_model, d_v * n_head)
        self.fc = nn.Linear(d_v * n_head, d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        # Q,K,V: [batch_size, len_q/k/v, d_model] mask: [batch_size, len_q, len_k]
        query = self.w_qs(query).view(  # [batch_size, n_head, len_q, d_k]
            -1, query.size(1), self.n_head, self.d_k).transpose(1, 2)
        key = self.w_ks(key).view(  # [batch_size, n_head, len_k, d_k]
            -1, key.size(1), self.n_head, self.d_k).transpose(1, 2)
        value = self.w_vs(value).view(  # [batch_size, n_head, len_v, d_v]
            -1, value.size(1), self.n_head, self.d_v).transpose(1, 2)
        if mask is not None: mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len|1, seq_len]
        x, attn = self.attention(query, key, value, mask=mask)  # [batch_size, n_head, len_v, d_v]
        x = x.transpose(1, 2)  # [batch_size, len_v, n_head, d_v]
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # [batch_size, len_v, n_head*d_v]
        x = self.dropout(self.fc(x))  # [batch_size, len_v, d_model]
        return (x, attn) if return_attn else x  # [batch_size, len_v, d_model]


class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_2(torch.relu(self.w_1(x))))


class AddAndNorm(nn.Module):
    """ Add & Norm """

    def __init__(self, d_model):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sub):
        return self.norm(x + sub(x))


class EncodeLayer(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_ffn, dropout=0.1):
        super(EncodeLayer, self).__init__()
        self.atn = MultiHeadAttention(
            d_model, n_head, d_k, d_k, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ffn, dropout=dropout)
        self.add_1 = AddAndNorm(d_model)
        self.add_2 = AddAndNorm(d_model)

    def forward(self, enc_input, enc_mask=None):
        enc_o = self.add_1(
            enc_input, lambda x: self.atn(x, x, x, enc_mask))
        return self.add_2(enc_o, self.ffn)


class DecodeLayer(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_ffn, dropout=0.1):
        super(DecodeLayer, self).__init__()
        self.atn_1 = MultiHeadAttention(
            d_model, n_head, d_k, d_k, dropout=dropout)
        self.atn_2 = MultiHeadAttention(
            d_model, n_head, d_k, d_k, dropout=dropout)
        self.ffn = FeedForward(d_model, d_ffn, dropout=dropout)
        self.add_1 = AddAndNorm(d_model)
        self.add_2 = AddAndNorm(d_model)
        self.add_3 = AddAndNorm(d_model)

    def forward(self, dec_i, enc_o, slf_mask=None, enc_mask=None):
        dec_o = self.add_1(
            dec_i, lambda dec_x: self.atn_1(dec_x, dec_x, dec_x, mask=slf_mask))
        dec_o = self.add_2(
            dec_o, lambda dec_x: self.atn_2(dec_x, enc_o, enc_o, mask=enc_mask))
        return self.add_3(dec_o, self.ffn)


class PositionalEncoding(nn.Module):

    def __init__(self, d_feature, n_position=256):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_feature)
        )

    @classmethod
    def _denominator(cls, d_feature):
        def _value(hid_j):
            return torch.tensor(1) / torch.pow(torch.tensor(10000), torch.tensor(2 * (hid_j // 2) / d_feature))
        return torch.tensor([_value(hid_j) for hid_j in range(d_feature)]).view(1, -1)

    @classmethod
    def _position(cls, n_position):
        return torch.arange(n_position).unsqueeze(-1)

    @classmethod
    def _sinusoid_table(cls, n_position, d_feature):
        return cls._position(n_position) * cls._denominator(d_feature)

    @classmethod
    def _get_sinusoid_encoding_table(cls, n_position, d_feature):
        """Sinusoid position encoding table."""
        sinusoid_table = cls._sinusoid_table(n_position, d_feature)
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])
        return sinusoid_table.unsqueeze(0)  # [1, n_position, d_feature]

    def forward(self, x):
        return x + self.position_table[:, :x.size(1)].clone().detach()


def pad_mask(x, pad_idx=0):
    # [batch_size, seq_len] -> [batch_size, 1, seq_len]
    return (x != pad_idx).unsqueeze(-2)


def subsequent_mask(x):
    """ 对角掩码矩阵 """
    # [batch_size, seq_len] -> [1, seq_len, seq_len]
    return (torch.tril(torch.ones((1, x.size(1), x.size(1)), device=x.device))).bool()


def sos(batch_size, sos_idx, device=None):  # [batch_size, 1]
    """
    初始化decoder输入数据
    """
    return torch.tensor(
        [sos_idx] * batch_size).unsqueeze(-1).to(device)


if __name__ == '__main__':
    n = PositionalEncoding(16, 100)
    xi = torch.zeros((1, 10, 16))
    print(n(xi))
