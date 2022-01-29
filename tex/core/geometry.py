import torch


def box_split(box, keepdim=False):
    assert box.size(-1) == 4  # dim=-1 [x y w h]
    rect_tuple = (
        box.index_select(-1, torch.tensor(0, device=box.device)),
        box.index_select(-1, torch.tensor(1, device=box.device)),
        box.index_select(-1, torch.tensor(2, device=box.device)),
        box.index_select(-1, torch.tensor(3, device=box.device))
    )
    return rect_tuple if keepdim \
        else (i.squeeze(-1) for i in rect_tuple)


def center_distance(box_a, box_b, is_sqrt=False):
    """ 计算矩形间的中心点距离 """
    a_x, a_y, a_w, a_h = box_split(box_a)
    b_x, b_y, b_w, b_h = box_split(box_b)
    d_x = (a_x + a_w / 2) - (b_x + b_w / 2)
    d_y = (a_y + a_h / 2) - (b_y + b_h / 2)
    dist = torch.pow(d_x, 2) + torch.pow(d_y, 2)
    return torch.sqrt(dist) if is_sqrt else dist


def diag_length(box, is_sqrt=False):
    """ 矩形对角线长度 """
    _, _, box_w, box_h = box_split(box, keepdim=False)
    dist = torch.pow(box_w, 2) + torch.pow(box_h, 2)
    return torch.sqrt(dist) if is_sqrt else dist


def distance(p1, p2, is_sqrt=False):  # [..., d]
    dist = torch.sum(torch.pow(p1 - p2, 2), dim=-1)
    return torch.sqrt(dist) if is_sqrt else dist


def r2p(box):
    box_x, box_y, box_w, box_h = box_split(box, keepdim=True)
    return torch.cat(
        (box_x, box_y, box_x + box_w, box_y + box_h), dim=-1)


def aspect_ratio(box):
    _, _, box_w, box_h = box_split(box, keepdim=False)
    return box_w / box_h


def area(box):
    _, _, box_w, box_h = box_split(box, keepdim=False)
    return box_w * box_h


def intersect(box_a, box_b):
    """
    Compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [..., A, 4].
      box_b: (tensor) bounding boxes, Shape: [..., B, 4].
    Return:
      (tensor) intersection area, Shape: [..., A, B].
    """

    A = box_a.size()[:-2], box_a.size()[-2]
    B = box_b.size()[:-2], box_b.size()[-2]

    a_x0, a_y0, a_x1, a_y1 = box_split(r2p(box_a), keepdim=True)
    b_x0, b_y0, b_x1, b_y1 = box_split(r2p(box_b), keepdim=True)

    a_min = torch.cat((a_x0, a_y0), dim=-1)
    b_min = torch.cat((b_x0, b_y0), dim=-1)
    a_max = torch.cat((a_x1, a_y1), dim=-1)
    b_max = torch.cat((b_x1, b_y1), dim=-1)

    max_xy = torch.min(
        a_max.unsqueeze(-2).expand(*A[0], A[1], B[1], 2),
        b_max.unsqueeze(-3).expand(*B[0], A[1], B[1], 2)
    )
    min_xy = torch.max(
        a_min.unsqueeze(-2).expand(*A[0], A[1], B[1], 2),
        b_min.unsqueeze(-3).expand(*B[0], A[1], B[1], 2)
    )

    inter = torch.clamp((max_xy - min_xy), min=0)
    inter_w = inter.index_select(
        -1, torch.tensor(0, device=inter.device)).squeeze(-1)
    inter_h = inter.index_select(
        -1, torch.tensor(1, device=inter.device)).squeeze(-1)

    return inter_w * inter_h


def jaccard(box_a, box_b):
    """
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [..., A, 4]
        box_b: (tensor) Prior boxes from prior box layers, Shape: [..., B, 4]
    Return:
        jaccard overlap: (tensor) Shape: [..., A, B]
    """
    inter = intersect(box_a, box_b)

    _, _, a_w, a_h = box_split(box_a, keepdim=False)
    _, _, b_w, b_h = box_split(box_b, keepdim=False)

    area_a = (a_w * a_h).unsqueeze(-1).expand_as(inter)
    area_b = (b_w * b_h).unsqueeze(-2).expand_as(inter)

    return inter / (area_a + area_b - inter)


def iou(box_a, box_b):
    return torch.diagonal(jaccard(box_a, box_b), dim1=-1, dim2=-2)


def ssi(box, contain_self=False):
    """ 矩形集合与自身相交的面积总和 """
    return torch.sum(
        torch.triu(intersect(box, box), 0 if contain_self else 1))


def mbr(*args):
    """ MBR 最小外接矩形 inputs:[..., 4] """
    box = torch.stack(args, -2) if len(args) > 1 else args[0]
    x0, y0, x1, y1 = box_split(r2p(box), keepdim=False)
    min_x = torch.min(x0, dim=-1).values
    min_y = torch.min(y0, dim=-1).values
    max_x = torch.max(x1, dim=-1).values
    max_y = torch.max(y1, dim=-1).values
    return torch.stack(
        (min_x, min_y, max_x - min_x, max_y - min_y), dim=-1)


if __name__ == '__main__':
    a = torch.tensor([[[90, 100, 100, 130], [100, 100, 100, 100], [40, 90, 100, 100]]], dtype=torch.float64)
    b = torch.tensor([[[100, 100, 125, 125], [100, 100, 125, 125], [100, 100, 100, 200]]], dtype=torch.float64)

    print(mbr(a, b))