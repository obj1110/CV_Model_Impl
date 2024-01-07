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

def main():
    # use GPU seems too slow in my PC, currenly do not knoe why
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([
            # RandomHorizontalFlip 是一个数据增强技术，它以一定的概率水平地翻转图像。这是一种常见的方法，用于增加训练深度学习模型时的数据多样性，有助于提高模型对不同方向图像的泛化能力。
            # RandomCrop 是另一种数据增强技术，它从图像中随机裁剪出一个指定大小的区域。这对于训练具有平移不变性要求的模型非常有用，因为它通过从原始图像中提取不同区域来增加训练数据的多样性。
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # os.getcwd()表示获取当前文件所在的路径
    # .. means one layer upper
    # ../.. means two layer upper
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    print(data_root)
    # 路径拼接
    image_path = os.path.join(data_root, "data", "flowers_data")  # flower data set path
    print(image_path)
    assert os.path.exists(image_path), f"指定的路径不存在: {image_path}"

    # 获取图片并且进行预处理
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    print(train_num)  # 3306


    flower_list = train_dataset.class_to_idx # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    cla_dict = dict((val, key) for key, val in flower_list.items()) # {0:'daisy', 1:'dandelion', 2:'roses', 3:'sunflower', 4:'tulips'}
    json_str = json.dumps(cla_dict, indent=4) # write cla_dict into json file
    # save the json_str to local
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # load Data
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    # this may take a long time to proceed
    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # HYPER_PARAMETER
    epochs = 20
    save_path = '../pthFile/AlexNet.pth'
    best_acc = -1.0
    train_steps = len(train_loader)

    # if reached ideal precision, can early stop
    for epoch in range(epochs):
        # net.train()表示启用dropout方法, only need during training process
        net.train()
        running_loss = 0.0
        # show a tqdm bar in console
        train_bar = tqdm(train_loader, file=sys.stdout)
        # why use the train_bar
        for step, data in enumerate(train_bar):
            images, labels = data
            # clear grad data before
            optimizer.zero_grad()
            # WARNING: it costs a lot of time here
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            # back-pro to every node
            loss.backward()
            # use optmizer to update data
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # after training, we began to evaluate
        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        # 禁止pytorch对梯度进行跟踪，也就是验证过程中pytorch不会去跟踪损失梯度
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                # 分为图片和对应的标签
                val_images, val_labels = val_data
                # 将图片传递到设备上
                outputs = net(val_images.to(device))

                # outputs的每一行的最大值
                # outputs就是[batch_size, num_classes]
                # 也就对应了样本以及可能的分类结果
                # 最后的结果就是，最大值和索引，[1]就是把索引拿出来
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

# this is crucial for multi-thread process
if __name__ == '__main__':
    main()