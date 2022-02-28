import torch
import torch.nn as nn
import tex.models.transformer.attention as attention


class BertEmbedding(nn.Module):

    def __init__(self, n_vocab, n_position=512, d_embedding=128, dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.position = nn.Embedding(n_position, d_embedding)
        self.segment = nn.Embedding(2, d_embedding)  # 0 or 1
        self.token = nn.Embedding(n_vocab, d_embedding, padding_idx=0)
        self.norm = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence, segment_label=None):
        # sequence.size(-1) batch_size, [seq_len]
        if segment_label is None:
            segment_label = torch.zeros(
                sequence.size(-1), dtype=torch.long, device=sequence.device)
        position_input = torch.arange(
            0, sequence.size(-1), dtype=torch.long, device=sequence.device)
        return self.dropout(
            self.norm(
                self.token(sequence) + self.position(
                    position_input) + self.segment(segment_label)
            )
        )


class ALBert(nn.Module):
    """ A Lite Bert """

    def __init__(self, n_vocab, d_embedding, d_model, n_head, d_k, d_ffn,
            n_shared_layer=12, n_position=512, pad_idx=0, dropout=0.1):
        super(ALBert, self).__init__()
        self.embedding = BertEmbedding(n_vocab, n_position, d_embedding, dropout)
        self.embedding_mapping = nn.Linear(d_embedding, d_model)
        self.shared_layer = attention.EncodeLayer(d_model, n_head, d_k, d_ffn, dropout)
        self.pad_idx, self.n_shared_layer = pad_idx, n_shared_layer

    def forward(self, sequence, segment_label=None):
        mask = attention.pad_mask(sequence, self.pad_idx)
        output = self.embedding(sequence, segment_label)
        output = self.embedding_mapping(output)
        for _ in range(self.n_shared_layer):
            output = self.shared_layer(output, mask)
        return output  # [batch_size, n_position, d_model]


class ALBertForNSP(nn.Module):

    def __init__(self, n_vocab, d_embedding, d_model, n_head, d_k, d_ffn,
            n_shared_layer=12, n_position=512, pad_idx=0, dropout=0.1):
        super(ALBertForNSP, self).__init__()
        self.bert = ALBert(n_vocab, d_embedding, d_model, n_head, d_k, d_ffn,
            n_shared_layer=n_shared_layer, n_position=n_position, pad_idx=pad_idx, dropout=dropout)
        self.cls_linear = nn.Linear(d_model, 1)

    def forward(self, sequence, segment_label=None):
        x = self.bert(sequence, segment_label)
        x = x.index_select(1, torch.tensor(0, device=sequence.device))  # pos:0|[CLS] [bs, 1, dim]
        return self.cls_linear(x.squeeze(1)).squeeze(1)  # [bs]


if __name__ == '__main__':
    nsp = ALBertForNSP(n_vocab=30522, d_embedding=128, d_model=768, n_head=12, d_k=64, d_ffn=3072, n_shared_layer=12)
    # nsp = ALBertForNSP(n_vocab=30522, d_embedding=128, d_model=4096, n_head=64, d_k=64, d_ffn=16384, n_shared_layer=12)
    print('model parameters:', sum(x.numel() for x in nsp.parameters()))
    # x = torch.randint(5000, [3, 512], dtype=torch.long)
    # x = nsp(x)
    # print(x.size())
