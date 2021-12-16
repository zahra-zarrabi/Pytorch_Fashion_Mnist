# Fashion MNIST Classifier
We proposed a CNN network for Fashion Mnist classification. We used framework pytorch==1.8.2.
## Dataset
Fashion MNIST image dataset includes a training set of 60,000 samples and a test set of 10,000 samples. Each sample is a 28x28 image on a gray scale associated with a tag of 10 classes.

## Train
To train the model, please run fallowing:
```
python3 train.py --device cuda
```

## Download weights
Please download the weights from [here](https://drive.google.com/file/d/11NlkMfSGvcGcR_0AuBCc1HPFZW4PJoNF/view?usp=sharing)  
## Test
To test the model, please run fallowing:
```
python test.py --device cuda
```
## Inference
To inference the model, please run fallowing:
```
python inference.py --device cuda --image_path data/2.jpeg
```

