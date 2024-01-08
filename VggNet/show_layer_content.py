import torch
import torch.nn as nn

from model import vgg

# vgg11 = VGG("vgg11", num_classes=1000, init_weights=False)
# print(vgg11)

model_name = "vgg11"
net = vgg(model_name=model_name, num_classes=5, init_weights=True)
print(net)