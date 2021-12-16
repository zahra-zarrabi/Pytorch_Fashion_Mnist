from Model import cnn_model
import torch
import torchvision
import argparse
from tqdm import tqdm

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--device', default='cpu', type=str)
args=my_parser.parse_args()

def calc_acc(preds,labels):
  _,pred_max=torch.max(preds,1)
  acc=torch.sum(pred_max==labels.data, dtype=torch.float64) / len(preds)
  return acc

device = torch.device(args.device)
model=cnn_model()
model = model.to(device)

batch_size=64
epochs=10
lr=0.01

#data preparing
transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.ConvertImageDtype(torch.float32),
                                          torchvision.transforms.RandomHorizontalFlip()
])
train_data=torchvision.datasets.FashionMNIST('./dataset',train=True,download=True,transform=transform)
train_data_loader=torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)

# compile
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
loss_function=torch.nn.CrossEntropyLoss()

#train
for epoch in range(epochs):
  print(f'Epoch:{epoch}')
  model.train(True)
  train_loss = 0.0
  train_acc = 0.0
  for images, labels in tqdm(train_data_loader):
    images = images.to(device)
    labels = labels.to(device)
    optimizer.zero_grad()

    preds_train = model(images)

    loss_train = loss_function(preds_train, labels)  # loss_train
    loss_train.backward()

    optimizer.step()

    train_loss += loss_train
    train_acc += calc_acc(preds_train, labels)

  total_loss = train_loss / len(train_data_loader)
  total_acc = train_acc / len(train_data_loader)
  print(f"loss_train:{total_loss},accuracy_train:{total_acc}")

torch.save(model.state_dict(), "models/fashion_mnist.pth")
