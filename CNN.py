#----------------------------
#  ConvolutionNeuralNetwork :
#----------------------------
import datasets
import device
import torch
import torch.nn as nn
from numpy import size
from scipy.special.cython_special import y1
from torch.utils.data import dataloader
from torch.version import cuda
from torchvision.transforms import Lambda, ToTensor
from transformers import PreTrainedModel


class ConvolutionNeuralNetwork(nn.module):
    def __init__(self):
        super(ConvolutionNeuralNetwork , self).__init__()

        self.conv = nn.conv2d(1, 16 , kernel_size= 3)
        self.relu = nn.ReLU()
        self.maxPool = nn.MaxPool2d(6)
        self.fc = nn.linear(16, 10 )

    def forward (self , x):
        x = self.MaxPool(self.relu(self.conv(x)))
        x = x.view(size(0) , -1)
        x = x.fc(x)
        return x

model = ConvolutionNeuralNetwork().to(device)
print(model)

#----------------------------
#  ReCurrentNeuralNetwork :
#----------------------------
class RecurrentNeuralNetwork(nn.module):
    def __init__(self):
        super(RecurrentNeuralNetwork ,self).__init__()

        self.rnn = nn.LSTM(input_seze = 10 , hidden_size = 20 , batch_first = True)
        self.fc  = nn.linear(20 , 1 )

    def forward ( self ,x):
        out , none = self.rnn(x)
        out        = self.fc(out[: , -1 , :])
        return out
model    = RecurrentNeuralNetwork().to(device)
print(model)

#------------------------------
#  Transformers :
#------------------------------

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

#------------------------------
#  GPU Training (Speed Boost) :
#------------------------------

device = torch.device("cuda " if torch.cuda.is_available() else "cpu")

model = ConvolutionNeuralNetwork().to(device)
x1 = x1.to(device)
y1 = y1.to(device)


#------------------------------
#  MyCNN_model:
#------------------------------

class RNNModel(nn.module):
    def __init__(self):
        super(RNNModel , self).__init__()

        self.layer1 = nn.Linear( 10 , 20)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear( 20 , 20 )
        self.fc     = nn.Linear( 20, 4)

    def forward(self , x):
        x = self.layer2(self.relu(self.layer1(x)))
        x = self.fc(x)
        return x

#------------------------------
#  Transfer Learning :
#------------------------------
from torchvision import models, datasets

model = model.resnet50(pretrained= True)

for param in model.parameter():
    param.requires_grad = False

model.fc = nn.Linear( 510 , 10)

#----------------------------------
#  Optimizing the Model Parameters :
#----------------------------------
Learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameter() , lr = Learning_rate )

def train(criterion , model, optimizer , dataloder):
    sixe = len(dataloader.dataset)
    model.train()

    for Batch,(x, y) in enumerate(dataloader):
        x , y = x.to(device) , y.to(device)

        #compute predict and loss
        predict = model(x)
        losss = criterion( predict , y )

        #backpropagation
        optimizer.zero_grad()
        losss.backward()
        optimizer.step()

        if Batch % 10 == 0 :
            losss, current = losss.item() , (Batch+1)* len(x)
            print (f"Batch: {Batch+1} ,loss: {losss} , Current: {current} ")


























