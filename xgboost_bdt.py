#!/usr/bin/env python
# coding: utf-8

# import relevant libraries
import os, sys, io, datetime, pickle, copy

import numpy as np
import matplotlib.pyplot as plt

from keras import layers, models, optimizers, initializers
from keras import models
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json

import pandas as pd
import tensorflow as tf
from xgboost import XGBRegressor
from tqdm import tqdm

import numpy as np
import pandas as pd

from tensorboard_utils import *


allfeatures=10000
features = ["dsid", "eventnumber", "feat_etmiss", "feat_etmiss_phi", "feat_lep_pt", "feat_lep_phi", "feat_lep_eta", "feat_lep_e", "feat_jet1_pt", "feat_jet1_phi", "feat_jet1_eta", "feat_jet1_e", "feat_jet1_btagged", "feat_jet2_pt", "feat_jet2_phi", "feat_jet2_eta", "feat_jet2_e", "feat_jet2_isbtagged", "feat_jet3_pt", "feat_jet3_phi", "feat_jet3_eta", "feat_jet3_e", "feat_jet3_isbtagged", "feat_jet4_pt", "feat_jet4_phi", "feat_jet4_eta", "feat_jet4_e", "feat_jet4_isbtagged", "feat_jet5_pt", "feat_jet5_phi", "feat_jet5_eta", "feat_jet5_e", "feat_jet5_isbtagged"]
df_a= pd.read_csv('createImages/results.csv/mc15_13TeV.410000.PwPyEG_P2012_ttbar_hdamp172p5_nonallhad.1lep.csv',names=features, header=0, usecols=range(0,len(features)))#, nrows=1000)
df_a['target'] = 2
df_b = pd.read_csv('createImages/results.csv/mc15_13TeV.410010.PwPyEG_P2012_singletop_tchan_lept.1lep.csv',names=features, header=0, usecols=range(0,len(features)))#, nrows=1000)
df_b['target'] = 1
df_c = pd.read_csv('createImages/results.csv/mc15_13TeV.999998.Sh_221_NNPDF30NNLO_Wjets.1lep.csv',names=features, header=0, usecols=range(0,len(features)))#, nrows=1000)
df_c['target'] = 0

# Add dataframes
df = df_a.append(df_b)
df = df.append(df_c)

# Add njets and nbjets columns
jets = df.columns[df.columns.str.endswith("pt")]
bjets = df.columns[df.columns.str.endswith("btagged")]
col = df.loc[:,jets]
df['feat_njets'] = 5 - (col == 0).sum(axis=1) 
col = df.loc[:,bjets]
df['feat_bjets'] = 5 - (col == 0).sum(axis=1)

# Shuffle dataframe and Divide dataset
df.sample(frac=1)
df['rand'] = np.random.randint(1, 11, df.shape[0])
df['type'] = df['rand'].apply(lambda x: 'val' if x > 9 else 'train')

features = df.columns[df.columns.str.startswith("feat_")]
target = 'target'

model = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=500, n_jobs=-1, colsample_bytree=0.1, objective="multi:softmax", num_class=3)
model.fit(df[df.type == "train"][features], df[df.type == "train"][target])

# Save Model
import pickle
pickle.dump(model, open('XGBoost_model.dat', "wb"))

from sklearn.metrics import confusion_matrix
predictions = model.predict(df[df.type == "val"][features])

# Commented out IPython magic to ensure Python compatibility.
import seaborn as sn
import matplotlib.pyplot as plt
# %matplotlib inline

y_true = df[df.type == "val"][target].values
cm = confusion_matrix(y_true , predictions)
print(cm)
classes = ['Wjets','tchannel','ttbar']

conf_matrix_norm = plot_confusion_matrix(cm=cm,classes=classes, normalize=True, tensor_name='Confusion Matrix Normalized',saveImg=True)

sys.exit()

#classes = ['ttbar','sgtop','wjets']
tick_marks = np.arange(len(classes))
df_cm = pd.DataFrame(cm, index = classes,
                  columns = classes)
df_cm = df_cm.div(df_cm.sum(axis=1), axis=0)
fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1)

ax.set_xlabel('Predicted', fontsize=12)
ax.set_xticks(tick_marks)
ax.set_xticklabels(classes, fontsize=12, va ='center')
ax.xaxis.set_label_position('bottom')
ax.xaxis.tick_bottom()

ax.set_ylabel('True Label', fontsize=12)
ax.set_yticks(tick_marks)
ax.set_yticklabels(classes, fontsize=12, va ='center')
ax.yaxis.set_label_position('left')
ax.yaxis.tick_left()

#sn.set(fontsize=12) # for label size
sn.heatmap(df_cm, annot=True, cmap='Oranges', ax=ax) # font size
plt.show()
  
fig.set_tight_layout(True)
#fig.set_canvas(plt.gcf().canvas)
fig.savefig('confusion_matrix_xgboost.png')   # save the figure to file

from sklearn.metrics import classification_report
print(classification_report(y_true, predictions, target_names=classes))
