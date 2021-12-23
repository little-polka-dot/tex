
TEX_STRUCTURE_SETTINGS = {
    'pad_idx': 0,
    'enc_d_input': 1,  # encoder输入图层数
    'enc_d_model': 512,  # encoder输出图层数
    'enc_layers': [3, 4, 6, 3],
    'enc_block': 'BasicBlock',
    'enc_n_position': 4096,  # 须大于等于图像卷积后的size
    'dec_d_model': 512,  # decoder维度
    'dec_n_vocab': 9,  # 表结构描述语言词汇量
    'dec_seq_len': 128,  # decoder序列长度
    'dec_n_head': 8,
    'dec_d_k': 128,
    'dec_h_layers': 5,
    'dec_d_layers': 1,
    'dec_dropout': 0.1,
    'dec_n_position': 256,  # 须大于等于dec_seq_len
}
