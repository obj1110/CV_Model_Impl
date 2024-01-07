import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet

device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# pre-process the image
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image from dataset and show
# from PIL import Image
img_path = "../data/demo1.jpg"
img = Image.open(img_path)
# img.show()

# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
# torch.Size([1, 3, 224, 224])
print(img.shape)

# just for display text for flower image
# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r") as f:
    class_indict = json.load(f)

# create model
model = AlexNet(num_classes=5).to(device)

# load model weights
weights_path = "../pthFile/AlexNet.pth"
model.load_state_dict(torch.load(weights_path))

# Enter EVAL mode, without dropout of neurons
model.eval()
# without tracking the gradient of forward tensors
with torch.no_grad():
    # predict class
    output = torch.squeeze(model(img.to(device))).cpu()
    # use softmax to convert from logit to probability
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)], predict[predict_cla].numpy())
plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], predict[i].numpy()))
plt.show()
