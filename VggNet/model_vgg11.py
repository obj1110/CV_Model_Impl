import torch
import torch.nn as nn

# stride 和 padding 如果都是1，那么输出的高和宽是不变的
class VGG11(nn.Module):
    def __init__(self, num_classes = 1000, init_weights = False):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            # [224, 224, 3] to [224， 224, 64]
            nn.Conv2d(3, 64, kernel_size=3, padding = 1, stride=1),
            nn.ReLU(inplace = True),
            # [224, 224, 64] to [112, 112, 64]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [112,112,64] to [112,112,128]
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            # [112,112,128] to [56, 56, 128]
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._init_weights()

    def forward(self, x):
        # N * 3 * 224 * 224
        x = self.features(x)
        x = torch.flatten(x, start_dim = 1)
        return self.classifier(x)


    # nn.init.xavier_uniform and nn.init.xavier_uniform_ differ in that if they will replace and modify the input tensor in place
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                # 使用Xavier均匀初始化方法来初始化卷积层的权重
                nn.init.xavier_normal_(m.weight)
                # 检查卷积层是否有偏置项
                if m.bias is not None:
                    # 将偏置项设置为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


# Attention for the 6th and 8th layer, 11th and 13th layer, 16th and 18th layer
# VGG(
#   (features): Sequential(
#     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU(inplace=True)
#     (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): ReLU(inplace=True)
#     (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): ReLU(inplace=True)
#     (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (9): ReLU(inplace=True)
#     (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (12): ReLU(inplace=True)
#     (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (14): ReLU(inplace=True)
#     (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#     (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (17): ReLU(inplace=True)
#     (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (19): ReLU(inplace=True)
#     (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=5, bias=True)
#   )
# )

