import torch
import torch.nn.functional as F
import tex.core.geometry as geo


def iou_loss(output, target, ignore_zero=True):
    """ IoU 输入： [seq_len, 4] """
    if ignore_zero:
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = geo.iou(output, target)
    loss = 1 - iou  # -torch.log(iou) inf会导致模型无法拟合
    return torch.mean(loss)


def distance_iou_loss(output, target, ignore_zero=True):
    """ DIoU 输入： [seq_len, 4] """
    if ignore_zero:
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = geo.iou(output, target)
    mbr_diag = geo.diag(geo.mbr(output, target))
    dist_center = geo.center_distance(output, target)
    loss = 1 - iou + dist_center / mbr_diag
    return torch.mean(loss)


def complete_iou_loss(output, target, ignore_zero=True):
    """ CIoU 输入： [seq_len, 4] """
    if ignore_zero:
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = geo.iou(output, target)
    mbr_diag = geo.diag(geo.mbr(output, target))
    dist_center = geo.center_distance(output, target)
    t_asp = torch.arctan(geo.aspect_ratio(target))
    p_asp = torch.arctan(geo.aspect_ratio(output))
    value = torch.pow(
        t_asp - p_asp, 2) * (4 / (torch.pi * torch.pi))
    alpha = torch.div(value, ((1 - iou) + value))
    loss = 1 - iou + dist_center / mbr_diag + alpha * value
    return torch.mean(loss)


def score_complete_iou_loss(output, target, ignore_zero=True):
    """ Score CIoU 输入： [seq_len, 4] """
    if ignore_zero:
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]
    iou = geo.iou(output, target)
    mbr_diag = geo.diag(geo.mbr(output, target))
    dist_center = geo.center_distance(output, target)
    t_asp = torch.arctan(geo.aspect_ratio(target))
    p_asp = torch.arctan(geo.aspect_ratio(output))
    value = torch.pow(
        t_asp - p_asp, 2) * (4 / (torch.pi * torch.pi))
    alpha = torch.div(value, ((1 - iou) + value))
    loss = 1 - iou + dist_center / mbr_diag + alpha * value
    return torch.sum(torch.softmax(loss, -1) * loss)


def tile_iou_loss(output, target, ignore_zero=True, progress_exp=10):
    """
    输入： [seq_len, 4] 暂不支持batch_size维度
      f = (sum(intersect(X, X)) + | sum(X) - mbr(X) |) / mbr(X)
      ...
    增大progress_exp会延后模型训练中坐标修正产生作用的时机
    """
    if ignore_zero:
        output = output[(target > 0).any(-1)]
        target = target[(target > 0).any(-1)]

    iou = geo.iou(output, target)
    mbr_diag = geo.diag(geo.mbr(output, target))
    dist_center = geo.center_distance(output, target)
    t_asp = torch.arctan(geo.aspect_ratio(target))
    p_asp = torch.arctan(geo.aspect_ratio(output))
    value = torch.pow(
        t_asp - p_asp, 2) * (4 / (torch.pi * torch.pi))
    alpha = torch.div(value, ((1 - iou) + value))
    loss = 1 - iou + dist_center / mbr_diag + alpha * value
    loss = torch.sum(torch.softmax(loss, -1) * loss)

    p_mbr, t_mbr = geo.mbr(output), geo.mbr(target)
    p_tile = geo.sum_si(output) + torch.abs(
        geo.area(p_mbr) - torch.sum(geo.area(output)))
    t_tile = geo.sum_si(target) + torch.abs(
        geo.area(t_mbr) - torch.sum(geo.area(target)))
    tile_value = torch.abs(
        p_tile / geo.area(p_mbr) - t_tile / geo.area(t_mbr))

    tile_value = 1 - 2 * torch.arctan(tile_value) / torch.pi

    # tile_value = torch.pow(
    #     precision_bn, torch.div(-1, tile_value))
    # tile_value = torch.div(
    #     1 - tile_value, 1 + tile_value)  # (0, 1]

    iou_pr = torch.sum(torch.softmax(-iou, -1) * iou)
    iou_pr = torch.pow(iou_pr, progress_exp)  # 修正比率

    # print(iou_pr.item(), tile_value.item())

    return loss + 1 - iou_pr * tile_value


def cls_loss(output, target, pad_idx=0, smoothing=0.01, weight=None):
    return F.cross_entropy(output, target.to(torch.long),
        ignore_index=pad_idx, label_smoothing=smoothing, weight=weight)


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
                   ignore_zero=True, pad_idx=0, smoothing=0.01, weight=None):
    """ 如果输入box为(x,y,w,h)格式 则设置is_transform为True """
    # outputs tuple([batch_size, seq_len, dim], [batch_size, seq_len, 4])
    # targets tuple([batch_size, seq_len], [batch_size, seq_len, 4])
    cls_output, box_output = outputs
    cls_target, box_target = targets
    cls_loss_value = batch_mean(cls_loss, cls_output, cls_target,
        pad_idx=pad_idx, smoothing=smoothing, weight=weight)
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
        torch.tensor([[[0.1, 0.1, 0.1, 0.2], [0.1, 0.2, 0.1, 0.11], [0.1, 0.3, 0.1, 0.11]]], dtype=torch.float64)
        # torch.tensor([[[0.1, 0.1, 0.1, 0.3], [0.1, 0.1, 0.1, 0], [0.1, 0.1, 0.1, 0]]], dtype=torch.float64)
    )
    print(a[0].argmax(-1))
    b = (
        torch.tensor([[7, 4, 0]]),
        torch.tensor([[[0.1, 0.1, 0.1, 0.1], [0.1, 0.2, 0.1, 0.1], [0.1, 0.3, 0.1, 0.1]]], dtype=torch.float64)
    )
    print(complete_iou_loss(a[1], b[1]))
    print(score_complete_iou_loss(a[1], b[1]))
    print(tile_iou_loss(a[1], b[1]))