import torch


def sequence_accuracy(output, target, ignore_idx=0):
    """
    总体评估一个模型 如果需要评估每一个类的识别情况则需要使用f1值来评估
    准确率 预测正确的占总的比重
    output [batch_size, seq_len, n_classes]
    target [batch_size, seq_len]
    """
    acc = (torch.argmax(output, dim=-1) == target).to(torch.float)
    return torch.nanmean(
        acc.masked_fill(target == ignore_idx, torch.nan))


def sequence_precision(output, target, use_idx=0):
    """
    查准率 正确预测为正的占全部预测为正的比例
    output [batch_size, seq_len, n_classes]
    target [batch_size, seq_len]
    """
    pre = torch.argmax(output, dim=-1)
    pre_matrix = pre.masked_fill(
        pre == use_idx, 1).masked_fill(pre != use_idx, 0)
    tar_matrix = pre_matrix.masked_fill(target != use_idx, 0)
    return torch.sum(tar_matrix) / torch.sum(pre_matrix)


def sequence_recall(output, target, use_idx=0):
    """
    查全率 正确预测为正的占全部实际为正的比例
    output [batch_size, seq_len, n_classes]
    target [batch_size, seq_len]
    """
    pre = torch.argmax(output, dim=-1)
    tar_matrix = target.masked_fill(
        target == use_idx, 1).masked_fill(target != use_idx, 0)
    pre_matrix = tar_matrix.masked_fill(pre != use_idx, 0)
    return torch.sum(pre_matrix) / torch.sum(tar_matrix)


def sequence_f1(output, target, use_idx=0):
    """ F1值(H-mean) """
    p = sequence_precision(output, target, use_idx)
    r = sequence_recall(output, target, use_idx)
    return 2 * p * r / (p + r)


if __name__ == '__main__':
    x = torch.tensor([[[-0.9317, -0.8505,  1.4930],
         [ 0.4288, -0.5441, -0.9881],
         [ 1.5186, -0.2999, -0.0654],
         [ 0.3426,  0.7747,  0.5787],
         [ 0.0668,  0.1664, -0.0880],
         [-1.3781, -0.7853,  0.7559]],

        [[ 0.6923,  0.7525,  1.1177],
         [-0.2764, -0.0344,  0.6761],
         [ 1.0452,  0.3951,  1.9040],
         [-0.7490, -0.7552, -0.2190],
         [-0.0197, -0.5762,  0.1544],
         [-0.1455, -0.2509, -0.5170]]])
    t = torch.tensor([[1,0,0,1,1,2],[2,2,2,2,1,0]])
    print(sequence_recall(x, t, 2))
    print(sequence_precision(x, t, 2))
    print(sequence_f1(x, t, 2))

