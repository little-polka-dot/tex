import torch
import torch.nn.functional as F
import tex.core.geometry as geo


def iou_loss(output, target, ignore_zero=True):
    if ignore_zero:  # 不支持batch_size维度
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = geo.iou(target, output)
    loss = 1 - iou  # -torch.log(iou) inf会导致模型无法拟合
    return torch.mean(loss)


def distance_iou_loss(output, target, ignore_zero=True):
    """ DIoU """
    if ignore_zero:  # 不支持batch_size维度
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = geo.iou(target, output)
    mbr_diag = geo.diag_length(geo.mbr(target, output))
    d_center = geo.center_distance(target, output)
    loss = 1 - iou + d_center / mbr_diag
    return torch.mean(loss)


def complete_iou_loss(output, target, ignore_zero=True):
    """ CIoU """
    if ignore_zero:  # 不支持batch_size维度
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = geo.iou(target, output)
    mbr_diag = geo.diag_length(geo.mbr(target, output))
    d_center = geo.center_distance(target, output)
    asp_tar = torch.arctan(geo.aspect_ratio(target))
    asp_pre = torch.arctan(geo.aspect_ratio(output))
    value = torch.pow(
        asp_tar - asp_pre, 2) * (4 / (torch.pi * torch.pi))
    alpha = value / ((1 - iou) + value)  # 完全重合时该值为nan
    loss = 1 - iou + d_center / mbr_diag + alpha * value
    return torch.mean(loss)


def tile_penalty(box_a, box_b):
    """
    f = 矩形集合的重叠面积和 + 矩形集合最小外接矩形与矩形集合面积和的差值
    return: | f(a) - f(b) |
    """
    # TODO: 计算过程中是否需要归一化?
    a_mbr, a_ssi = geo.mbr(box_a), geo.ssi(box_a)
    b_mbr, b_ssi = geo.mbr(box_b), geo.ssi(box_b)
    a_tile = a_ssi + torch.abs(
        geo.area(a_mbr) - torch.sum(geo.area(box_a)))
    b_tile = b_ssi + torch.abs(
        geo.area(b_mbr) - torch.sum(geo.area(box_b)))
    return torch.abs(a_tile - b_tile)


def tile_iou_loss(output, target, ignore_zero=True):
    if ignore_zero:  # 不支持batch_size维度
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = geo.iou(target, output)
    mbr_diag = geo.diag_length(geo.mbr(target, output))
    d_center = geo.center_distance(target, output)
    asp_tar = torch.arctan(geo.aspect_ratio(target))
    asp_pre = torch.arctan(geo.aspect_ratio(output))
    value = torch.pow(
        asp_tar - asp_pre, 2) * (4 / (torch.pi * torch.pi))
    alpha = value / ((1 - iou) + value)  # 完全重合时该值为nan
    loss = 1 - iou + d_center / mbr_diag + alpha * value
    return torch.mean(loss + tile_penalty(output, target))


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


def structure_loss(outputs, targets,
                   ignore_zero=True, pad_idx=0, smoothing=0.1, weight=None):
    """ 如果输入box为(x,y,w,h)格式 则设置is_transform为True """
    # outputs tuple([batch_size, seq_len, dim], [batch_size, seq_len, 4])
    # targets tuple([batch_size, seq_len], [batch_size, seq_len, 4])
    cls_output, box_output = outputs
    cls_target, box_target = targets
    cls_loss_value = batch_mean(
        cls_loss, cls_output, cls_target, pad_idx=pad_idx, smoothing=smoothing, weight=weight)
    iou_loss_value = batch_mean(
        tile_iou_loss, box_output, box_target, ignore_zero=ignore_zero)
    return cls_loss_value, iou_loss_value


if __name__ == '__main__':
    a = (
        torch.tensor([[[1.4949, 0.7972, -0.3455, -0.4040, 1.2417, 0.4645, 0.1462,
                  1.9950, 1.3542],
                 [-0.0285, -0.2565, -0.0992, 0.0920, 0.8295, 0.3249, 0.1341,
                  -0.4668, -1.3706],
                 [1.2456, -0.5448, -0.5127, -0.3453, 0.6549, -0.1191, -0.4428,
                  -0.4353, -0.4258]]]),
        torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]]], dtype=torch.float64)
    )
    print(a[0].argmax(-1))
    b = (
        torch.tensor([[7, 4, 0]]),
        torch.tensor([[[0.1, 0.1, 0.125, 0.2], [0.1, 0.1, 0.125, 0.2], [0.1, 0.1, 0.125, 0.2]]], dtype=torch.float64)
    )

    print(iou_loss(a[1], b[1]))
    print(distance_iou_loss(a[1], b[1]))
    print(complete_iou_loss(a[1], b[1]))
    print(tile_iou_loss(a[1], b[1]))
