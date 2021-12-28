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


def box_transformer(
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


def iou_loss(outputs, targets, is_transformer=True):
    """
    如果坐标为x,y,w,h格式 则将is_transformer设置为True
    """
    for batch, target in enumerate(targets):
        output = outputs[batch]  # [seq_len, 4]
        if is_transformer:
            target = box_transformer(target)  # [seq_len, 4]
            output = box_transformer(output)  # [seq_len, 4]
        iou = torch.diag(
            jaccard(target_masked(target), output))  # [seq_len]
        yield torch.nanmean(-torch.log(iou))


def cls_loss(outputs, targets, pad_idx=0, smoothing=0.1):
    for batch, target in enumerate(targets):
        output = outputs[batch]
        yield F.cross_entropy(
            output, target, ignore_index=pad_idx, label_smoothing=smoothing)


def structure_loss(
        outputs, targets, is_transformer=True, pad_idx=0, smoothing=0.1):
    # outputs tuple([batch_size, seq_len, dim], [batch_size, seq_len, 4])
    # targets tuple([batch_size, seq_len], [batch_size, seq_len, 4])
    cls_output, box_output = outputs
    cls_target, box_target = targets
    cls_loss_value = list(cls_loss(cls_output, cls_target, pad_idx, smoothing))
    cls_loss_value = sum(cls_loss_value) / len(cls_loss_value)
    iou_loss_value = list(iou_loss(box_output, box_target, is_transformer))
    iou_loss_value = sum(iou_loss_value) / len(iou_loss_value)
    return cls_loss_value, iou_loss_value


if __name__ == '__main__':
    a = (
        torch.randn([1, 3, 9]),
        torch.tensor([[[1, 1, 2, 2], [2, 2, 2, 2], [1, 1, 1, 1]]], dtype=torch.float64)
    )
    print(a[0].size(), a[1].size())
    b = (
        torch.randint(9, [1, 3]),
        torch.tensor([[[1, 1, 2, 2], [2, 2, 6, 6], [1, 1, 0, torch.nan]]], dtype=torch.float64)
    )
    print(b[0].size(), b[1].size())
    print(structure_loss(a, b))
