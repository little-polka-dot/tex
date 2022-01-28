import torch


def accuracy(output, target, ignore_idx=0):
    """
    总体评估一个模型 如果需要评估每一个类的识别情况则需要使用f1值来评估
    准确率 预测正确的占总的比重
    output [..., seq_len, n_classes]
    target [..., seq_len]
    """
    acc = torch.argmax(output, dim=-1) == target
    acc = torch.where(acc, 1.0, 0.0)
    return torch.mean(acc[target != ignore_idx])


def precision(output, target, use_idx=0):
    """
    查准率 正确预测为正的占全部预测为正的比例
    output [..., seq_len, n_classes]
    target [..., seq_len]
    """
    mask = torch.argmax(output, dim=-1) != use_idx
    pre_matrix = torch.where(mask, 0, 1)
    tar_matrix = pre_matrix.masked_fill(target != use_idx, 0)
    return torch.sum(tar_matrix) / torch.sum(pre_matrix)


def recall(output, target, use_idx=0):
    """
    查全率 正确预测为正的占全部实际为正的比例
    output [..., seq_len, n_classes]
    target [..., seq_len]
    """
    mask = torch.argmax(output, dim=-1) != use_idx
    tar_matrix = torch.where(target == use_idx, 1, 0)
    pre_matrix = tar_matrix.masked_fill(mask, 0)
    return torch.sum(pre_matrix) / torch.sum(tar_matrix)


def h_mean(output, target, use_idx=0):
    """ F1 | H-mean """
    p = precision(output, target, use_idx)
    r = recall(output, target, use_idx)
    return 2 * p * r / (p + r)


f1 = h_mean


if __name__ == '__main__':
    x = torch.tensor([[[-0.9317, -0.8505, 1.4930],
                       [0.4288, -0.5441, -0.9881],
                       [1.5186, -0.2999, -0.0654],
                       [0.3426, 0.7747, 0.5787],
                       [0.0668, 0.1664, -0.0880],
                       [-1.3781, -0.7853, 0.7559]],

                      [[0.6923, 0.7525, 1.1177],
                       [-0.2764, -0.0344, 0.6761],
                       [1.0452, 0.3951, 1.9040],
                       [-0.7490, -0.7552, -0.2190],
                       [-0.0197, -0.5762, 0.1544],
                       [-0.1455, -0.2509, -0.5170]]])
    t = torch.tensor([[1, 0, 0, 1, 1, 2], [2, 2, 2, 2, 1, 0]])
    print(accuracy(x, t, 0))
    print(recall(x, t, 3))
    print(precision(x, t, 2))
    print(f1(x, t, 2))
