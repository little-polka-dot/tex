import numpy as np


def gauss_noise(image, loc=0, scale=0.1):
    """
        loc: 平均值
        scale: 标准差
    """
    return np.uint8(
        np.clip(np.array(image / 255) + np.random.normal(
            loc, scale, image.shape), 0, 1) * 255)


if __name__ == '__main__':
    img = np.random.randint(0, 255, (3, 5))
    print(img)
    img = gauss_noise(img)
    print(img)
