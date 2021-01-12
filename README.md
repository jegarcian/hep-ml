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

## Artemisa Setup

To run on artemisa :

   * Setup to load ROOT to create images 

```console
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
setupATLAS;lsetup "root 6.18.04-x86_64-centos7-gcc8-opt" --quiet
```
   * Setup to run training :
 
```console
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install keras=2.3.1
pip install tensorflow==1.15
pip install sklearn
pip install matplotlib
pip install tfplot
pip install tensorflow-plot
```

once it has been done once, it only would need the activate part of the the settings. 
