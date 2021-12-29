from tex.datasets.labels import StructLang
import random
import numpy as np


def square_padding(image, padding=0):
    """ 矩形图片填充为正方形 """
    height, width = image.shape[:2]
    new_size = max(height, width)
    return cv2.copyMakeBorder(
        image,
        int((new_size - height) / 2),
        int((new_size - height + 1) / 2),
        int((new_size - width) / 2),
        int((new_size - width + 1) / 2),
        cv2.BORDER_CONSTANT, value=padding
    )


def gauss_noise(image, loc=0, scale=0.01):
    """ loc: 平均值 scale: 标准差 """
    return getattr(np, str(image.dtype))(np.clip(
        image / 255 + np.random.normal(loc, scale, image.shape), 0, 1) * 255)


def structure_transform(cfg, x_data: np.ndarray, y_data: dict):

    description = StructLang.from_object(y_data['description'])
    cut_len = random.randint(1, description.size + 1)  # 随机截断表格描述语句(用于模型训练)
    seq_labels = description.labels(cfg['seq_len'], False, True, cut_len)
    seq_inputs = description.labels(cfg['seq_len'], True, False, cut_len)

    if cfg['normalize_position']:  # 坐标归一化
        seq_position = [
            [
                i[0]/x_data.shape[1],  # x / W
                i[1]/x_data.shape[0],  # y / H
                i[2]/x_data.shape[1],  # w / W
                i[3]/x_data.shape[0],  # h / H
            ] for i in y_data['position']
        ]
    else:
        seq_position = y_data['position']

    if cfg['gauss_noise']:  # 图片添加噪音
        x_data = gauss_noise(
            x_data, cfg['gauss_noise']['loc'], cfg['gauss_noise']['scale'])

    if not cfg['flim_mode']:
        x_data = np.ones_like(x_data, np.uint8) * 255 - x_data  # 对正常的图片进行颜色翻转

    # TODO: 二值化 膨胀腐蚀前还是后?

    if cfg['erosion_dilation']:  # 膨胀腐蚀
        kernel = np.ones(cfg['erosion_dilation']['kernel'], np.uint8)
        for _ in range(cfg['erosion_dilation']['iterations']):
            x_data = cv2.erode(cv2.dilate(x_data, kernel=kernel), kernel=kernel)

    x_data = square_padding(x_data) / (255 if cfg['normalize_image'] else 1)  # 正方形填充并归一化

    return (x_data, seq_inputs), (seq_labels, seq_position)


if __name__ == '__main__':
    import cv2
    x = cv2.imread('F:/img.png')[:-1]
    print(x.dtype)
    print(x.shape)
    x = np.ones_like(x) * 255 - x
    x = square_padding(gauss_noise(x))
    k = np.ones([5, 5], np.uint8)
    x = cv2.erode(cv2.dilate(x, kernel=k), kernel=k)
    print(x.shape)
    cv2.imshow('img', x)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    np.ndarray()
