import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
image_path = os.path.join(data_root, "data", "flowers_data_threeFolder")  # flower data set path
assert os.path.exists(image_path), f"指定的路径不存在: {image_path}"

# 获取图片并且进行预处理
test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                     transform=data_transform)
test_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=1, shuffle=True,
                                           num_workers=0)

json_path = "./class_indices.json"
with open(json_path, "r") as f:
    class_indict = json.load(f)
print(class_indict)

# create model
model = AlexNet(num_classes=5).to(device)
weights_path = "../pthFile/AlexNet.pth"
model.load_state_dict(torch.load(weights_path))

model.eval()
acc = 0.0
# without tracking the gradient of forward tensors
with torch.no_grad():
    test_bar = tqdm(test_loader, file=sys.stdout)
    for val_data in test_bar:
        # 分为图片和对应的标签
        val_image, val_label = val_data
        # 将图片传递到设备上
        outputs = model(val_image.to(device))

        # outputs的每一行的最大值
        # outputs就是[batch_size, num_classes]
        # 也就对应了样本以及可能的分类结果
        # 最后的结果就是，最大值和索引，[1]就是把索引拿出来
        predict_y = torch.max(outputs, dim=1)[1]
        print(predict_y, val_label)
        acc += torch.eq(predict_y, val_label.to(device)).sum().item()
        print(acc)
val_accurate = acc / test_num
print(val_accurate) # 0.695054945054945