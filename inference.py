import torch
import torchvision
import argparse
from Model import cnn_model
import cv2
import numpy as np

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--device',default=torch.device('cpu'),type=str)
my_parser.add_argument('--image',type=str)
args=my_parser.parse_args()

target=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.ConvertImageDtype(torch.float32),
                                          torchvision.transforms.RandomHorizontalFlip()
])

device = torch.device(args.device)
model=cnn_model()
model = model.to(device)

model.load_state_dict(torch.load('fashion_mnist.pth',map_location=torch.device(args.device)))

model.eval()

img = cv2.imread(args.image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (28, 28))
tensor = transform(img).unsqueeze(0).to(device)

# process
pred = model(tensor)

# postprocess
pred = pred.cpu().detach().numpy()
pred = np.argmax(pred)
output = target[pred]

print('prediction:',output)