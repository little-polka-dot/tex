import torch
import torch.nn.functional as F
import tex.core.geo as geo


def iou_loss(output, target, ignore_zero=True):
    if ignore_zero:
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = torch.diag(geo.jaccard(target, output))
    # -torch.log(iou) inf会导致模型无法拟合
    return torch.mean(1 - iou)


def distance_iou_loss(output, target, ignore_zero=True):
    """ DIoU """
    if ignore_zero:
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = torch.diag(geo.jaccard(target, output))
    drc = geo.center_distance(target, output)
    lnd = geo.diag_length(
        geo.min_enclosing_rect(target, output))
    # TODO 取最大值还是取平均值?
    return torch.mean(1 - iou + drc / lnd)


def complete_iou_loss(output, target, ignore_zero=True):
    """ CIoU """
    if ignore_zero:
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = torch.diag(geo.jaccard(target, output))
    drc = geo.center_distance(target, output)  # 矩形中心距离
    lnd = geo.diag_length(
        geo.min_enclosing_rect(target, output))
    ast = torch.arctan(geo.aspect_ratio(target))
    aso = torch.arctan(geo.aspect_ratio(output))
    con = (4 / (torch.pi * torch.pi))  # 4/pi^2
    val = con * torch.pow(ast - aso, 2)
    aph = val / ((1 - iou) + val)  # 完全重合时该值为nan
    return torch.mean(1 - iou + drc / lnd + aph * val)


def cls_loss(output, target, pad_idx=0, smoothing=0.1, weight=None):
    return F.cross_entropy(  # 内部会自动调用softmax
        output, target.to(torch.long), ignore_index=pad_idx, label_smoothing=smoothing, weight=weight)


def batch_mean(loss_func, outputs, targets, **kwargs):
    # TODO: 循环效率较低 需要优化
    return torch.mean(
        torch.stack(
            [
                loss_func(
                    outputs[batch], targets[batch], **kwargs)
                for batch in range(targets.size(0))
            ]
        )
    )


def structure_loss(
        outputs, targets, ignore_zero=True, pad_idx=0, smoothing=0.1, weight=None):
    """ 如果输入box为(x,y,w,h)格式 则设置is_transform为True """
    # outputs tuple([batch_size, seq_len, dim], [batch_size, seq_len, 4])
    # targets tuple([batch_size, seq_len], [batch_size, seq_len, 4])
    cls_output, box_output = outputs
    cls_target, box_target = targets
    cls_loss_value = batch_mean(
        cls_loss, cls_output, cls_target, pad_idx=pad_idx, smoothing=smoothing, weight=weight)
    iou_loss_value = batch_mean(
        complete_iou_loss, box_output, box_target, ignore_zero=ignore_zero)
    return cls_loss_value, iou_loss_value


if __name__ == '__main__':
    a = (
        torch.randn([1, 3, 9]),
        torch.tensor([[[0.1, 0.1, 0.2, 0.2], [0.2, 0.2, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]], dtype=torch.float64)
    )
    print(a[0].size(), a[1].size())
    b = (
        torch.randint(9, [1, 3]),
        torch.tensor([[[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.3, 0.3], [0.0, 0.0, 0.0, 0.0]]], dtype=torch.float64)
    )
    print(b[0].dtype, b[1].dtype)
    print(structure_loss(a, b)[1])
