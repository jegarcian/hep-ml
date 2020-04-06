# hep-ml

Repository to perform an study similar to the documented at :

	   https://arxiv.org/abs/1708.07034

## Getting started

Download the code :

```
git clone https://github.com/jegarcian/hep-ml.git
cd hep-ml
git clone https://github.com/dgurkaynak/tensorflow-cnn-finetune.git
cp transfer-learning/* tensorflow-cnn-finetune/resnet/
```

## Create Images

Directory createImages contains code to read ntuples and create images as output. Instructions on how to run it are available in the README

## Running Training

### Keras example

Simple model using keras 

```
python simple.py
```


### Running Custom CNN or Transfer Learning

Use keras to run different types of NN

```
python run_model.py -m [model]
```

### Transfer Learning

Finetuning of commonly used neural nets with Tensorflow. See directory :

```
tensorflow-cnn-finetune/resnet/
```

for examples.

## Docker Image

A dockerfile containing needed packages is available at ```DockerFile```.
