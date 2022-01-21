import torch
import torch.nn as nn
from typing import Type, Union
from tex.utils.functional import any_gt, mul, all_odd, map_, list_
from tex.models.transformer import attention


class Block(nn.Module):

    expansion = 1

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(Block, self).__init__()
        self.subnetwork = sub
        self.downsample = None
        if any_gt(stride, 1) or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion, (1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        r = x  # id(r) == id(x)
        if callable(self.downsample):
            r = self.downsample(r)
        x = self.net(x)
        if callable(self.subnetwork):
            x = self.subnetwork(x)
        return torch.relu(r + x)
        

class BasicBlock(Block):

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(BasicBlock, self).__init__(in_planes, planes, stride, sub)
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, planes, (3, 3), padding=(1, 1), stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(in_planes),
        )

        
class BottleNeck(Block):

    expansion = 4

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(BottleNeck, self).__init__(in_planes, planes, stride, sub)
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, planes, (1, 1)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, (3, 3), padding=(1, 1), stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, (1, 1)),
            nn.BatchNorm2d(in_planes),
        )


class CoTAttention(nn.Module):

    def __init__(self, d_model, d_hidden, kernel_size, stride=(1, 1)):
        super(CoTAttention, self).__init__()
        assert all_odd(kernel_size)  # 卷积尺度必须为奇数
        padding = list_(map_(lambda x: (x - 1) // 2, kernel_size))
        self.kernel_size = kernel_size
        self.key_mapping = nn.Sequential(  # TODO: conv groups=4 ?
            nn.Conv2d(d_model, d_model, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )
        self.val_mapping = nn.Sequential(
            nn.Conv2d(d_model, d_model, (1, 1), bias=False),
            nn.BatchNorm2d(d_model)
        )
        self.atn_mapping = nn.Sequential(
            nn.Conv2d(2 * d_model, d_hidden, (1, 1), bias=False),
            nn.BatchNorm2d(d_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_hidden, d_model * mul(kernel_size), (1, 1))
        )
        self.pool = None
        if any_gt(stride, 1):
            self.pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        if callable(self.pool): x = self.pool(x)  # bs,c,h,w
        k_1 = self.key_mapping(x)  # bs,c,h,w
        val = self.val_mapping(x).view(x.size(0), x.size(1), -1)  # bs,c,h*w
        atn = self.atn_mapping(torch.cat([k_1, x], dim=1))  # bs,c*alpha,h,w
        atn = atn.view(x.size(0), x.size(1), mul(self.kernel_size), x.size(2), x.size(3))
        atn = atn.mean(2, keepdim=False).view(x.size(0), x.size(1), -1)  # bs,c,h*w
        k_2 = torch.softmax(atn, dim=-1) * val  # bs,c,h*w
        return k_1 + k_2.view(*x.size())  # bs,c,h,w


class CoTBottleNeck(Block):

    expansion = 4

    def __init__(self, in_planes, planes, stride=(1, 1), sub=None):
        super(CoTBottleNeck, self).__init__(in_planes, planes, stride, sub)
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, planes, (1, 1)),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            CoTAttention(planes, planes, (3, 3), stride=stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, (1, 1)),
            nn.BatchNorm2d(in_planes),
        )


def make_layer(layers: int, block: Type[Block],
               in_planes, planes, stride=(1, 1), sub=None):
    return nn.Sequential(
        block(in_planes, planes, stride=stride, sub=sub),
        *[
            block(
                planes * block.expansion, planes, sub=sub
            ) for _ in range(1, layers)
        ]
    )


class ContextModeling(nn.Module):

    def __init__(self, channels):
        super(ContextModeling, self).__init__()
        self.score_conv = nn.Conv2d(channels, 1, (1, 1))

    def forward(self, x):  # x: [batch_size, c, h, w]
        score = self.score_conv(x).view(x.size(0), 1, -1)  # [batch_size, 1, h*w]
        score = torch.softmax(score, dim=-1).unsqueeze(-1)  # [batch_size, 1, h*w, 1]
        x = x.view(x.size(0), x.size(1), -1).unsqueeze(1)  # [batch_size, 1, c, h*w]
        return torch.matmul(x, score).transpose(1, 2)  # [batch_size, c, 1, 1]


class ContextTransformer(nn.Module):

    def __init__(self, channels, d_hidden):
        super(ContextTransformer, self).__init__()
        self.conv_1 = nn.Conv2d(channels, d_hidden, (1, 1))
        self.layer_norm = nn.LayerNorm(d_hidden)
        self.conv_2 = nn.Conv2d(d_hidden, channels, (1, 1))

    def forward(self, x):
        return self.conv_2(  # [batch_size, channels, 1, 1]
            torch.relu(  # [batch_size, d_hidden, 1, 1]
                self.layer_norm(
                    self.conv_1(x).view(x.size(0), -1))  # [batch_size, d_hidden]
            ).unsqueeze(-1).unsqueeze(-1)
        )


class GlobalContextBlock(nn.Module):

    def __init__(self, channels, d_hidden):
        super(GlobalContextBlock, self).__init__()
        self.net = nn.Sequential(
            ContextModeling(channels), ContextTransformer(channels, d_hidden))

    def forward(self, x): return x + self.net(x)


class BackbonePreprocess(nn.Module):

    def __init__(self, d_input, d_model):
        super(BackbonePreprocess, self).__init__()
        # TODO: diff between 7x7 and 3x3
        self.net = nn.Sequential(
            nn.Conv2d(d_input, d_model, (7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, x): return self.net(x)


class BackboneEncoder(nn.Module):

    def __init__(self, d_input, d_model,
                 block: Union[Block, str], layers, d_layer=(64, 128, 256, 512)):
        super(BackboneEncoder, self).__init__()
        if isinstance(block, str): block = globals()[block]
        self.pre_layer = BackbonePreprocess(d_input, d_layer[0])
        self.layers = nn.ModuleList([
            make_layer(
                layers[0],
                block,
                d_layer[0],
                d_layer[0],
                stride=(1, 1),
                sub=GlobalContextBlock(
                    d_layer[0] * block.expansion, d_layer[0] * block.expansion)
            ),
            make_layer(
                layers[1],
                block,
                d_layer[0] * block.expansion,
                d_layer[1],
                stride=(2, 2),
                sub=GlobalContextBlock(
                    d_layer[1] * block.expansion, d_layer[1] * block.expansion)
            ),
            make_layer(
                layers[2],
                block,
                d_layer[1] * block.expansion,
                d_layer[2],
                stride=(2, 2),
                sub=GlobalContextBlock(
                    d_layer[2] * block.expansion, d_layer[2] * block.expansion)
            ),
            make_layer(
                layers[3],
                block,
                d_layer[2] * block.expansion,
                d_layer[3],
                stride=(2, 2),
                sub=GlobalContextBlock(
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


class PositionalEncoder(nn.Module):

    def __init__(self, d_input, d_model, n_head, d_k, layers, dropout=0.1, d_ffn=None):
        super(PositionalEncoder, self).__init__()
        # TODO: 该模型不具有平移不变性与尺度不变性
        self.pos_mapping = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, kernel_size=(1, 1), bias=False),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.encoders = nn.ModuleList([
            attention.EncodeLayer(
                d_model, n_head, d_k, d_ffn=d_ffn, dropout=dropout) for _ in range(layers)
        ])

    def forward(self, x):
        m = self.encode_mask(x)
        x = self.pos_mapping(x)
        for layer in self.encoders:
            x = layer(x, m)
        # [batch_size, len, dim]
        return x, m  # mask to dec

    @staticmethod
    def encode_mask(x):
        # [batch_size, sql_len, d_input] -> [batch_size, 1, sql_len]
        return ((x > 0) | (x < 0)).any(-1).unsqueeze(-2)


if __name__ == '__main__':
    # x = torch.randn((3, 5, 4))
    # print(x)
    # s = x[:, :, 2:]
    # o = x[:, :, :2].repeat(1, 1, x.size(1))
    # w = x[:, :, :2].reshape(x.size(0), -1).unsqueeze(1).expand(-1, x.size(1), -1)
    # print(o)
    # print(w)
    # print(o - w)
    # x = torch.tensor([[[ 1.1538,  0.6114,  0.3825,  0.0084],
    #      [-0.6018, -0.5000,  1.6582,  1.3905],
    #      [-1.5990, -1.4760, -0.8017,  0.9766],
    #      [-0.4711,  0.5145, -0.9359,  2.0625],
    #      [-0.9621, -0.4437, -0.5028,  1.1662]],
    #
    #     [[-0.2074,  0.3908, -0.1805, -0.3430],
    #      [ 1.6591, -0.1458, -1.1184,  1.0134],
    #      [-1.4994, -0.1893,  0.6098, -0.5611],
    #      [ 0.6335,  0.9496,  0.5896,  0.0412],
    #      [ 0.3446,  0.5703,  2.0069, -0.3436]],
    #
    #     [[-0.0463, -0.0167,  0.2973, -0.7925],
    #      [ 1.1506,  1.7648,  0.6622,  2.0807],
    #      [-0.4990,  2.5495,  1.1142, -0.4095],
    #      [0, 0, 0, 0],
    #      [0,  0, 0,  0]]])
    # print(((x > 0) | (x < 0)).any(-1))
    pass