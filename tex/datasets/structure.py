from torch.utils.data import Dataset
import os
import cv2
import json


class SimpleStructureDataset(Dataset):

    def __init__(self, dataset_path, transform=None, grayscale=True):
        assert os.path.exists(dataset_path)
        self.x = os.path.join(dataset_path, 'X')
        assert os.path.exists(self.x)
        self.y = os.path.join(dataset_path, 'Y')
        assert os.path.exists(self.y)
        self.i = os.path.join(dataset_path, 'INDEX')
        assert os.path.exists(self.i)
        with open(self.i, 'r') as index_file:
            self.index = [tuple(
                i.split(':')) for i in index_file if i]
        self.transform = transform
        self.grayscale = grayscale

    def __getitem__(self, index):
        x_name, y_name = self.index[index]
        x_path = os.path.join(self.x, x_name.strip())
        assert os.path.exists(x_path)
        y_path = os.path.join(self.y, y_name.strip())
        assert os.path.exists(y_path)
        if self.grayscale:
            x_data = cv2.imread(x_path, cv2.IMREAD_GRAYSCALE)
        else:
            x_data = cv2.imread(x_path, cv2.IMREAD_COLOR)
        with open(y_path, 'r') as y_file:
            y_data = json.load(y_file)
        if callable(self.transform):
            x_data, y_data = self.transform(x_data, y_data)
        return x_data, y_data

    def add(self, x_data, y_data):
        x_name = f'X{len(self.index)}.png'
        x_path = os.path.join(self.x, x_name.strip())
        y_name = f'Y{len(self.index)}.json'
        y_path = os.path.join(self.y, y_name.strip())
        cv2.imwrite(x_path, x_data)
        with open(y_path, 'w') as y_file:
            json.dump(y_data, y_file)
        with open(self.i, 'a') as index_file:
            index_file.write(f'{x_name}:{y_name}\n')
        self.index.append((x_name, y_name))


if __name__ == '__main__':
    im = cv2.imread('F:/img1.png', cv2.IMREAD_GRAYSCALE)
    print(im.shape)
    # d = SimpleStructureDataset('E:/Code/Mine/github/tex/.data/train/structure')
    # for x in d: print(x[1])
    # print('*****************')
    # d.add(im, {1: 1, 2: 2})
    # for x in d: print(x[1])
