import torch


def sequence_accuracy(output, target, ignore_idx=0):
    """
    output [batch_size, seq_len, n_classes]
    target [batch_size, seq_len]
    """
    return torch.nanmean((torch.argmax(output, dim=-1) == target).to(
        torch.float).masked_fill(target == ignore_idx, torch.nan))


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
    t = torch.tensor([[2,0,0,1,1,2],[2,2,2,2,2,0]])
    print(sequence_accuracy(x, t))

