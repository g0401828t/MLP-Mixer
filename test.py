import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model import MlpMixer

import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="parameters")
parser.add_argument("--model", required=True, help="type in which model file you want to test")
args = parser.parse_args()

### parameters
model_name = "MlpMixer"
# path = "D:/projects"
path = os.path.dirname(os.getcwd()) # "D:/projects"
datapath = path + '/dataset'
modelpath = path + "/" + model_name + "/models/" + args.model
batch_size = 128


### 사용 가능한 gpu 확인 및 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)


"""
### test set for STL10
# from train_set
meanR, meanG, meanB = 0.4467106, 0.43980986, 0.40664646
stdR, stdG, stdB = 0.22414584, 0.22148906,  0.22389975
# define the image transforamtion for test_set
test_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
                transforms.Resize(224)
])
# load STL10 test dataset
test_set = datasets.STL10(datapath, split='test', download=True, transform=test_transformer)
print(test_set.data.shape)
# test_set loader
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# check dataloader
for x,y in test_loader:
    print(x.shape)
    print(y.shape)
    break
"""


### test set for Cifar10
test_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.229, 0.224, 0.225)),
                transforms.Resize(224)
])
test_set = datasets.CIFAR10(root=datapath, train=False,
                                       download=True, transform=test_transformer)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)





### model
# --------- base_model_param ---------
in_channels = 3
hidden_size = 768  # channels 512
num_classes = 10
patch_size = 16
resolution = 224
number_of_layers = 8
tokens_mlp_dim = 256
channels_mlp_dim = 2048
# ------------------------------------

model = MlpMixer(
    in_channels=in_channels,
    dim=hidden_size,
    num_classes=num_classes,
    patch_size=patch_size,
    image_size=resolution,
    depth=number_of_layers,
    tokens_mlp_dim=tokens_mlp_dim,
    channels_mlp_dim=channels_mlp_dim
)
model.to(device)



### test function
def test():
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    test_acc = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)

            # 순전파
            prediction = model(x)
            test_acc += (prediction.max(1)[1] == y).sum().item() * 100 / len(test_set)

    print("Acc: [{:.2f}%]".format(
        test_acc
    ))


### test
test()