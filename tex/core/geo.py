import torch


def intersect(box_a, box_b):  # box_a:[N, 4] box_b:[N, 4]
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
    a_min, a_max = box_a[:, :2], torch.stack(
        (box_a[:, 0] + box_a[:, 2], box_a[:, 1] + box_a[:, 3]), dim=1)
    b_min, b_max = box_b[:, :2], torch.stack(
        (box_b[:, 0] + box_b[:, 2], box_b[:, 1] + box_b[:, 3]), dim=1)
    max_xy = torch.min(
        a_max.unsqueeze(1).expand(box_a.size(0), box_b.size(0), 2),
        b_max.unsqueeze(0).expand(box_a.size(0), box_b.size(0), 2)
    )
    min_xy = torch.max(
        a_min.unsqueeze(1).expand(box_a.size(0), box_b.size(0), 2),
        b_min.unsqueeze(0).expand(box_a.size(0), box_b.size(0), 2)
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    # inter[:, :, 0] is the width of intersection and inter[:, :, 1] is height
    return inter[:, :, 0] * inter[:, :, 1]


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
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] * box_a[:, 3]).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = (box_b[:, 2] * box_b[:, 3]).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def iou(box_a, box_b): torch.diag(jaccard(box_a, box_b))


def min_enclosing_rect(box_a, box_b):
    """ 最小外接矩形(不考虑斜方向矩形的情况) """
    x_min = torch.min(box_a[:, 0], box_b[:, 0])
    x_max = torch.max(
        box_a[:, 0] + box_a[:, 2], box_b[:, 0] + box_b[:, 2])
    y_min = torch.min(box_a[:, 1], box_b[:, 1])
    y_max = torch.max(
        box_a[:, 1] + box_a[:, 3], box_b[:, 1] + box_b[:, 3])
    return torch.stack(
        (x_min, y_min, x_max - x_min, y_max - y_min), dim=1)


def center_distance(box_a, box_b, is_sqrt=False):
    """ 计算矩形间的中心点距离 """
    x_a = box_a[:, 0] + box_a[:, 2] / 2
    y_a = box_a[:, 1] + box_a[:, 3] / 2
    x_b = box_b[:, 0] + box_b[:, 2] / 2
    y_b = box_b[:, 1] + box_b[:, 3] / 2
    dist = torch.pow(x_a - x_b, 2) + torch.pow(y_a - y_b, 2)
    return torch.sqrt(dist) if is_sqrt else dist


def diag_length(box, is_sqrt=False):
    """ 矩形对角线长度 """
    dist = torch.pow(box[:, 2], 2) + torch.pow(box[:, 3], 2)
    return torch.sqrt(dist) if is_sqrt else dist


def aspect_ratio(box): return box[:, 2] / box[:, 3]  # 矩形宽高比


if __name__ == '__main__':
    a = torch.abs(torch.randn((10, 4)))
    b = torch.abs(torch.randn((10, 4)))
    x = center_distance(a, b)
