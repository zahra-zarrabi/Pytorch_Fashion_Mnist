import torch

class cnn_model(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn1 = torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2)
    self.cnn2=torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
    self.fc1=torch.nn.Linear(1568,10)

  def forward(self,x):
    x=self.cnn1(x)
    x=torch.relu(x)
    x=torch.nn.MaxPool2d(2, stride=2)(x)
    x=self.cnn2(x)
    x=torch.relu(x)
    x=torch.nn.MaxPool2d(2, stride=2)(x)
    x=x.reshape((x.shape[0],1568))
    x=self.fc1(x)
    return x