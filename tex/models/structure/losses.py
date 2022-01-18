import torch
import torch.nn.functional as F


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]
    # inter[:, :, 0] is the width of intersection and inter[:, :, 1] is height


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [A,4]
        box_b: (tensor) Prior boxes from prior box layers, Shape: [B,4]
    Return:
        jaccard overlap: (tensor) Shape: [A, B]
    """
    # input: (x_min, y_min, x_max, y_max)
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def box_transform(
        x, reverse=False, inplace=False):
    """
    reverse = False
        x, y, w, h -> x_min, y_min, x_max, y_max
    else
        x_min, y_min, x_max, y_max -> x, y, w, h
    """
    if not inplace: x = x.clone()
    if reverse:
        x[:, 2] = x[:, 2] - x[:, 0]
        x[:, 3] = x[:, 3] - x[:, 1]
    else:
        x[:, 2] = x[:, 2] + x[:, 0]
        x[:, 3] = x[:, 3] + x[:, 1]
    return x


def target_masked(x):
    """
    input: (x_min, y_min, x_max, y_max) 忽略无效面积的行
    [[0, 0, 0, 0]] -> [[nan, nan, nan, nan]]
    """
    area = (x[:, 3]-x[:, 1]) * (x[:, 2]-x[:, 0])
    mask = (area > 0).unsqueeze(-1).expand_as(x) == False
    return x.masked_fill(mask, torch.nan)


def bounding_rect(box_a, box_b):
    """
    最小外接矩形(不考虑斜方向矩形的情况)
    input: (x_min, y_min, x_max, y_max)
    """
    x_min = torch.min(box_a[:, 0], box_b[:, 0]).unsqueeze(1)
    y_min = torch.min(box_a[:, 1], box_b[:, 1]).unsqueeze(1)
    x_max = torch.max(box_a[:, 2], box_b[:, 2]).unsqueeze(1)
    y_max = torch.max(box_a[:, 3], box_b[:, 3]).unsqueeze(1)
    return torch.concat((x_min, y_min, x_max, y_max), dim=1)


def center_distance(box_a, box_b, is_sqrt=False):
    """
    计算矩形间的中心点距离
    input: (x_min, y_min, x_max, y_max)
    """
    x_a = (box_a[:, 0] + box_a[:, 2]) / 2
    y_a = (box_a[:, 1] + box_a[:, 3]) / 2
    x_b = (box_b[:, 0] + box_b[:, 2]) / 2
    y_b = (box_b[:, 1] + box_b[:, 3]) / 2
    dist = torch.pow(x_a - x_b, 2) + torch.pow(y_a - y_b, 2)
    return torch.sqrt(dist) if is_sqrt else dist


def diag_distance(x, is_sqrt=False):
    """
    矩形对角线长度
    input: (x_min, y_min, x_max, y_max)
    """
    dist = torch.pow(x[:, 0] - x[:, 2], 2) + torch.pow(x[:, 1] - x[:, 3], 2)
    return torch.sqrt(dist) if is_sqrt else dist


def aspect(x):
    """
    矩形宽高比
    input: (x_min, y_min, x_max, y_max)
    """
    return torch.abs(x[:, 0] - x[:, 2]) / torch.abs(x[:, 1] - x[:, 3])


def iou_loss(output, target, is_transform=True):
    """
    如果坐标为x,y,w,h格式 则将is_transform设置为True
    """
    if is_transform:
        target = box_transform(target)  # [seq_len, 4]
        output = box_transform(output)  # [seq_len, 4]
    iou = torch.diag(
        jaccard(target_masked(target), output))  # [seq_len]
    # yield torch.nanmean(-torch.log(iou))  # inf会导致模型无法拟合
    return torch.nanmean(1 - iou)


def distance_iou_loss(output, target, is_transform=True):
    """ DIoU """
    if is_transform:
        target = box_transform(target)  # [seq_len, 4]
        output = box_transform(output)  # [seq_len, 4]
    iou = torch.diag(
        jaccard(target_masked(target), output))  # [seq_len]
    dcn = center_distance(target, output)
    dbr = diag_distance(bounding_rect(target, output))
    # TODO 取最大值还是取平均值?
    return torch.nanmean(1 - iou + dcn / dbr)


def complete_iou_loss(output, target, is_transform=True):
    """ CIoU """
    if is_transform:
        target = box_transform(target)  # [seq_len, 4]
        output = box_transform(output)  # [seq_len, 4]
    iou = torch.diag(
        jaccard(target_masked(target), output))  # [seq_len]
    dcn = center_distance(target, output)  # 中心距离
    dbr = diag_distance(bounding_rect(target, output))  # 外接矩形对角距离
    val = torch.arctan(
        aspect(target)) - torch.arctan(aspect(output))
    val = (4 / (torch.pi * torch.pi)) * torch.pow(val, 2)
    aph = val / ((1 - iou) + val)  # 完全重合时该值为nan
    return torch.nanmean(1 - iou + dcn / dbr + aph * val)


def cls_loss(output, target, pad_idx=0, smoothing=0.1, weight=None):
    return F.cross_entropy(  # 内部会自动调用softmax
        output, target.to(torch.long), ignore_index=pad_idx, label_smoothing=smoothing, weight=weight)


def batch_mean(loss_func, outputs, targets, **kwargs):
    # TODO: 循环效率较低 需要优化
    return torch.nanmean(
        torch.stack(
            [
                loss_func(
                    outputs[batch], targets[batch], **kwargs)
                for batch in range(targets.size(0))
            ]
        )
    )


def structure_loss(
        outputs, targets, is_transform=True, pad_idx=0, smoothing=0.1, weight=None):
    # outputs tuple([batch_size, seq_len, dim], [batch_size, seq_len, 4])
    # targets tuple([batch_size, seq_len], [batch_size, seq_len, 4])
    cls_output, box_output = outputs
    cls_target, box_target = targets
    cls_loss_value = batch_mean(
        cls_loss, cls_output, cls_target, pad_idx=pad_idx, smoothing=smoothing, weight=weight)
    iou_loss_value = batch_mean(
        complete_iou_loss, box_output, box_target, is_transform=is_transform)
    return cls_loss_value, iou_loss_value


if __name__ == '__main__':
    # a = (
    #     torch.randn([1, 3, 9]),
    #     torch.tensor([[[1, 1, 2, 2.1], [2, 2, 2, 2], [1, 1, 1, 1]]], dtype=torch.float64)
    # )
    # print(a[0].size(), a[1].size())
    # b = (
    #     torch.randint(9, [1, 3]),
    #     torch.tensor([[[1, 1, 2, 2], [4, 4, 8, 10], [1, 1, 0, torch.nan]]], dtype=torch.float64)
    # )
    # print(b[0].size(), b[1].size())
    # print(structure_loss(a, b)[1])

    a = torch.tensor([[-1.1841e-01, -1.6656e-01, 6.5412e-01, -1.0629e+00, -1.3271e+00,
             1.0366e-01, -1.4296e-01, 2.8360e-01, 7.0585e-01],
            [7.0873e-02, -9.3102e-02, -1.5517e-01, -6.7302e-01, -1.2287e+00,
             -4.7391e-01, -8.0572e-01, 5.8163e-01, -2.8933e-01],
            [7.5680e-01, -4.0048e-01, -2.8725e-01, -1.1820e-01, -9.6922e-01,
             -7.6811e-01, -5.2617e-01, 3.7175e-01, -1.0800e+00],
            [3.7565e-01, -3.6742e-01, -2.4379e-01, -8.4440e-02, -8.4174e-01,
             -4.9515e-01, -3.2591e-01, 9.6110e-02, -5.4297e-01],
            [6.0946e-01, -5.9928e-02, 2.4187e-02, -4.1904e-02, -4.9279e-01,
             -3.8455e-01, -5.7985e-01, -2.1100e-01, -8.0949e-01],
            [5.8965e-01, -9.7454e-02, 3.0212e-01, 7.3201e-02, -5.2536e-01,
             -1.0408e+00, -3.0571e-01, -1.6644e-01, -1.0782e+00],
            [5.9076e-01, -4.1180e-01, 1.5714e-01, 4.7767e-01, -6.1096e-01,
             -4.9195e-01, -3.4107e-01, 3.1699e-01, -1.2260e+00],
            [9.6945e-01, -1.9266e-01, -2.6951e-01, -8.9753e-04, -4.2443e-01,
             -9.9056e-01, -6.4078e-01, 3.3390e-01, -8.4124e-01],
            [6.7456e-01, -1.0834e-01, -3.5329e-01, -2.7932e-02, -8.3893e-01,
             -5.3103e-01, -6.8160e-01, 6.7784e-02, -1.0258e+00],
            [8.4554e-01, 3.5428e-02, -3.7791e-01, 9.6975e-02, -3.3241e-01,
             -7.9069e-01, -2.4536e-01, 1.6838e-01, -7.4127e-01],
            [9.4695e-01, -1.5077e-01, 3.7117e-01, 1.7694e-01, -1.3474e-01,
             -8.9901e-01, -3.3682e-01, -1.4817e-01, -8.5693e-01],
            [7.6990e-01, -5.0259e-01, -2.1367e-01, 2.6977e-01, -4.2020e-01,
             -9.1868e-01, -3.8074e-01, 2.5671e-01, -8.3524e-01],
            [4.6722e-01, 2.5790e-01, -3.6349e-02, 1.5344e-01, -5.7409e-01,
             -2.9632e-01, 1.9248e-01, -4.3736e-01, -5.8371e-01],
            [-1.9299e-01, 7.1035e-01, -8.9198e-02, -4.6325e-01, -2.9522e-01,
             -1.8959e-01, -3.9859e-01, 2.9892e-01, -2.3609e-01],
            [1.9581e-01, 9.6004e-01, 4.6204e-02, -9.1585e-01, -6.2016e-01,
             2.1968e-01, 2.4252e-01, 2.2919e-01, -5.4120e-01],
            [-3.5649e-02, 7.5910e-01, -8.0415e-02, -4.8493e-01, -6.1986e-01,
             -6.9504e-01, -2.8432e-01, 2.8090e-01, -5.3881e-01],
            [-2.3806e-01, 1.6980e-01, 1.4028e-01, -5.0291e-01, -2.1068e-01,
             -5.1065e-01, -1.1921e-01, 1.7778e-02, 1.4234e-01],
            [4.0150e-01, 2.0715e-01, 5.0572e-02, -6.6514e-01, -9.3974e-01,
             -4.4237e-01, -5.0368e-02, 8.2389e-01, -6.0805e-01],
            [9.1964e-02, 9.4335e-02, -2.6350e-01, -5.9475e-01, -6.3158e-01,
             -3.8152e-01, -3.5223e-01, 4.6788e-01, -4.6243e-01],
            [1.2355e-01, -2.7521e-01, -1.3532e-01, -5.6983e-01, -2.7261e-01,
             -3.4614e-02, -3.4437e-01, 9.2892e-02, -6.2948e-01]],
           dtype=torch.float64)
    b = torch.tensor([3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0],
       dtype=torch.int32)
    print(cls_loss(a, b))