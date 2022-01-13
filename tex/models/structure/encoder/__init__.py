import torch
import torch.nn as nn
from tex.models.structure.encoder import blocks
from tex.models.structure.encoder import gc


class EncoderPreprocess(nn.Module):

    def __init__(self, d_input, d_model):
        super(EncoderPreprocess, self).__init__()
        # TODO: diff between 7x7 and 3x3
        self.net = nn.Sequential(
            nn.Conv2d(d_input, d_model, (7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(d_model, d_model, (1, 1), bias=False)
        )

    def forward(self, x): return self.net(x)


class Encoder(nn.Module):

    def __init__(self, d_input, d_model, block, layers, d_layer=(64, 128, 256, 512)):
        super(Encoder, self).__init__()
        if isinstance(block, str): block = getattr(blocks, block)
        self.pre_layer = EncoderPreprocess(d_input, d_layer[0])
        self.layers = nn.ModuleList([
            blocks.make_layer(
                layers[0],
                block,
                d_layer[0],
                d_layer[0],
                stride=(1, 1),
                sub=gc.GlobalContextBlock(
                    d_layer[0] * block.expansion, d_layer[0] * block.expansion)
            ),
            blocks.make_layer(
                layers[1],
                block,
                d_layer[0] * block.expansion,
                d_layer[1],
                stride=(2, 2),
                sub=gc.GlobalContextBlock(
                    d_layer[1] * block.expansion, d_layer[1] * block.expansion)
            ),
            blocks.make_layer(
                layers[2],
                block,
                d_layer[1] * block.expansion,
                d_layer[2],
                stride=(2, 2),
                sub=gc.GlobalContextBlock(
                    d_layer[2] * block.expansion, d_layer[2] * block.expansion)
            ),
            blocks.make_layer(
                layers[3],
                block,
                d_layer[2] * block.expansion,
                d_layer[3],
                stride=(2, 2),
                sub=gc.GlobalContextBlock(
                    d_layer[3] * block.expansion, d_layer[3] * block.expansion)
            ),
        ])
        self.layers_map = nn.ModuleList([
            nn.Conv2d(
                d_layer[1] * block.expansion, d_model, (1, 1), bias=False),
            nn.Conv2d(
                d_layer[2] * block.expansion, d_model, (1, 1), bias=False),
            nn.Conv2d(
                d_layer[3] * block.expansion, d_model, (1, 1), bias=False)
        ])

    def forward(self, x):
        """
        将细粒度，中粒度，粗粒度三个特征图合并为一个向量
        输出： [batch_size, img_len(取决于输入图像的size), d_model]
        """
        x = self.pre_layer(x)
        output = None
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx > 0:
                vec = self.layers_map[idx - 1](x)
                vec = vec.view(vec.size(0), vec.size(1), -1)
                if output is None:
                    output = vec
                else:
                    output = torch.cat((output, vec), -1)
        return output.transpose(1, 2)


if __name__ == '__main__':
    net = Encoder(1, 256, blocks.BasicBlock, [3, 4, 6, 3])
    i = torch.randn((10, 1, 224, 224))
    print(net(i).size())
