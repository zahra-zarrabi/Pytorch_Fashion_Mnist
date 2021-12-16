import torch
import torchvision
from tqdm import tqdm
import argparse
from Model import cnn_model

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--device',default=torch.device('cpu'),type=str)
args=my_parser.parse_args()

def calc_acc(preds,labels):
  _,pred_max=torch.max(preds,1)
  acc=torch.sum(pred_max==labels.data,dtype=torch.float64)/len(preds)
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
test_data=torchvision.datasets.FashionMNIST('./dataset',train=False,download=True,transform=transform)
test_data_loader=torch.utils.data.DataLoader(test_data,batch_size=batch_size,shuffle=False)

model.load_state_dict(torch.load('fashion_mnist.pth',map_location=torch.device(args.device)))

loss_function=torch.nn.CrossEntropyLoss()

model.eval()
test_loss = 0.0
test_acc = 0.0
for images, labels in tqdm(test_data_loader):
    images = images.to(device)
    labels = labels.to(device)

    preds_test = model(images)

    loss_test = loss_function(preds_test, labels)

    test_loss += loss_test
    test_acc += calc_acc(preds_test, labels)

total_loss = test_loss / len(test_data_loader)
total_acc = test_acc / len(test_data_loader)

print(f"loss_eval:{total_loss},accuracy_eval:{total_acc}")
