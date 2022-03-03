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
            n_layer=12, n_position=512, pad_idx=0, dropout=0.1):
        super(ALBert, self).__init__()
        self.embedding = BertEmbedding(n_vocab, n_position, d_embedding, dropout)
        if d_embedding == d_model:
            self.embedding_mapping = nn.Identity()  # d_embedding == d_hidden
        else:
            self.embedding_mapping = nn.Linear(d_embedding, d_model)
        self.shared_layer = attention.EncodeLayer(d_model, n_head, d_k, d_ffn, dropout)
        self.pad_idx, self.n_layer = pad_idx, n_layer

    def forward(self, sequence, segment_label=None, return_all=False):
        mask = attention.pad_mask(sequence, self.pad_idx)
        embedding = self.embedding(sequence, segment_label)
        layer_outputs = [self.embedding_mapping(embedding)]
        # output size: [batch_size, n_position, d_model]
        for _ in range(self.n_layer):
            layer_outputs.append(
                self.shared_layer(layer_outputs[-1], mask))
        return layer_outputs[1:] if return_all else layer_outputs[-1]


class ALBertForNSP(nn.Module):

    def __init__(self, n_vocab, d_embedding, d_model, n_head, d_k, d_ffn,
            n_layer=12, n_position=512, pad_idx=0, dropout=0.1):
        super(ALBertForNSP, self).__init__()
        self.bert = ALBert(n_vocab, d_embedding, d_model, n_head, d_k, d_ffn,
            n_layer=n_layer, n_position=n_position, pad_idx=pad_idx, dropout=dropout)
        self.pool = nn.Linear(d_model, d_model)
        self.last = nn.Linear(d_model, 1)

    def forward(self, sequence, segment_label=None):
        x = self.bert(sequence, segment_label)
        x = x.index_select(1, torch.tensor(0, device=sequence.device))  # [bs, 1, dim]
        x = torch.tanh(self.pool(x.squeeze(1)))
        return torch.sigmoid(self.last(x).squeeze(-1))  # [bs]


if __name__ == '__main__':
    model = ALBertForNSP(n_vocab=30000, d_embedding=128, d_model=768, n_head=12, d_k=64, d_ffn=3072, n_layer=12)
    # model = ALBertForNSP(n_vocab=30000, d_embedding=128, d_model=4096, n_head=64, d_k=64, d_ffn=16384, n_layer=12)
    # print('model parameters:', sum(x.numel() for x in model.parameters()))
    # x = torch.randint(30000, [3, 512], dtype=torch.long)
    # x = nsp(x)
    # print(x)
    print('bert/embeddings/word_embeddings:', sum(x.numel() for x in model.bert.embedding.token.parameters()))
    print('bert/embeddings/token_type_embeddings:', sum(x.numel() for x in model.bert.embedding.segment.parameters()))
    print('bert/embeddings/position_embeddings:', sum(x.numel() for x in model.bert.embedding.position.parameters()))
    print('bert/embeddings/LayerNorm:', sum(x.numel() for x in model.bert.embedding.norm.parameters()))
    print('bert/encoder/embedding_hidden_mapping_in:', sum(x.numel() for x in model.bert.embedding_mapping.parameters()))
    print('bert/encoder/transformer/attention/self:', sum(x.numel() for x in model.bert.shared_layer.slf_atn.parameters()))
    print('bert/encoder/transformer/LayerNorm_0:', sum(x.numel() for x in model.bert.shared_layer.res_slf.parameters()))
    print('bert/encoder/transformer/ffn/intermediate/dense/kernel:', sum(x.numel() for x in model.bert.shared_layer.fin_ffn.parameters()))
    print('bert/encoder/transformer/LayerNorm_1:', sum(x.numel() for x in model.bert.shared_layer.res_ffn.parameters()))
    print('bert/pooler/dense:', sum(x.numel() for x in model.pool.parameters()))

