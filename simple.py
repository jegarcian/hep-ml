#!/usr/bin/env python
# coding: utf-8

# import relevant libraries
import os, sys, io, datetime
import numpy as np

from keras import layers, models, optimizers, initializers
from keras import models
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from sklearn.metrics import classification_report

# Functions to save information for tensorboard
from tensorboard_utils import TrainValTensorBoard

bsize = 32
wsize = 224
hsize = 224

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# ====================================================================
#  Build and set up model
# ====================================================================
def build_model(xsize=224,ysize=224,channels=3,nlabels=2,lr=0.000001):
    model = models.Sequential()

    # input of shape: (image_height, image_weight, image_channels)
    # convolution layers
    model.add(layers.Conv2D(32, (7,7), activation='relu', input_shape=(xsize,ysize,channels)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (7,7), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (7,7), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))    

    # fully connected layers after 3D to 1D flat
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu',kernel_initializer=initializers.glorot_uniform(seed=0)))
    model.add(layers.Dropout(0.5)) # <- comprobar si es necesario

    model.add(layers.Dense(nlabels, activation='softmax'))
    
    # model.summary()
    
    model.compile(optimizer=optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# ====================================================================
#  main
# ====================================================================
def main():

    # parse options from command line
    from optparse import OptionParser
    parser = OptionParser(usage = "usage: %prog arguments", version="%prog")
    parser.add_option("-r", "--report",      dest="report",  action="store_true",    help="create report (default: %default)")
    parser.add_option("-w", "--weights",     dest="weights",                         help="load pre-made weights (default: %default)")
    parser.add_option("-l", "--lr",          dest="learning",                        help="set learning rate (default: %default)")
    parser.add_option("-b", "--batch",       dest="batch",                           help="set batch per epoch (default: %default)")
    parser.add_option("-e", "--epochs",      dest="epochs",                          help="number of epochs to run (default: %default)")
    parser.add_option("-v", "--verbose",      dest="verbose", action="store_true",   help="print more information (default: %default)")
    parser.set_defaults(verbose=True, report=False, weights=None, learning=0.0001, epochs=100, batch=None)
    (options,args) = parser.parse_args()
    # parser.print_help()

    now = datetime.datetime.now()
    logs_dir = now.strftime('./logs/%d%m%H%M')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))
    
    # create a data generator
    data_train = ImageDataGenerator()
    data_val = ImageDataGenerator()

    # Load images for training
    train_it = data_train.flow_from_directory(
        directory=r"/data/HEPML/InputImages/train",
        target_size=(wsize, hsize),
        color_mode="rgb",
        batch_size=bsize,
        class_mode="categorical",
        shuffle=True,
        seed=100
    )

    # Load images for validation
    val_it = data_val.flow_from_directory(
        directory=r"/data/HEPML/InputImages/val",
        target_size=(wsize, hsize),
        color_mode="rgb",
        batch_size=bsize, 
        class_mode="categorical",
        shuffle=False,
        seed=42
    )
    
    # Training and validating model
    model = build_model(lr=float(options.learning))

    if options.batch :
        STEP_SIZE_TRAIN=int(options.batch)
        STEP_SIZE_VALID=int(options.batch)
    else :
        STEP_SIZE_TRAIN=train_it.n//train_it.batch_size
        STEP_SIZE_VALID=val_it.n//val_it.batch_size
    
    if options.weights :
        try :
            print(" Loading weights : ",options.weights)  
            model.load_weights(options.weights)  
        except :
            print (" Error Loading weights")
            sys.exit()

    # Train the model
    tensorboard = TrainValTensorBoard(log_dir=logs_dir, write_graph=True, update_freq=500, val=val_it)

    history = model.fit_generator(train_it, epochs = int(options.epochs), steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_VALID, validation_data=val_it, callbacks=[tensorboard])
    
    # Save model training
    model_weights_path = now.strftime('simple_fc_model.%d%m%H%M.h5')
    model.save_weights(model_weights_path)

    
    if options.report :
        target_names = list(val_it.class_indices.keys())
        Y_pred = model.predict_generator(val_it, val_it.n//val_it.batch_size )
        y_pred = np.argmax(Y_pred, axis=1)
        print(classification_report(val_it.classes, y_pred, target_names=target_names))

        for x in range(0,len(y_pred)) :
            if val_it.classes[x] != y_pred[x] :
                print("File : ",val_it.filenames[x], str(val_it.classes[x]), str(y_pred[x]), str(Y_pred[x]))

    #from keras.models import load_model
    #pred = model.predict_generator(val_it, val_it.n // val_it.batch_size )
    #max_pred = np.argmax(pred, axis=1)
    #cnf_mat = confusion_matrix(val_it.classes, max_pred)
    #print (val_it.classes,pred,max_pred)
    #print (cnf_mat)

# ====================================================================
#  __main__
# ====================================================================
if __name__ == '__main__':
    main()
