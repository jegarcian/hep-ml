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
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json

import tensorflow as tf

from sklearn.metrics import classification_report

# Functions to save information for tensorboard
from tensorboard_utils import TrainValTensorBoard

bsize = 64
wsize = 224
hsize = 224

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


# ====================================================================
#  Get Classes to train
# ====================================================================
def getClasses(directory):
    
    classes = []
    dirs = os.listdir( directory)
    for d in dirs :
        if os.path.isdir(directory+"/"+d) :
            classes.append(d)
            
    #classes.sort(key=lambda v: v.upper())
    classes.sort()
    
    return classes



# ====================================================================
#  Tranfers Learning models
# ====================================================================
def build_model(xsize=224, ysize=224, channels=3, nlabels=2, lr=0.000001, bn=False, dropout=0.5, mtype='vgg'):

    from keras.applications.resnet50 import ResNet50
    from keras.applications.resnet import ResNet101
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg19 import VGG19
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.xception import Xception

    # Create the model
    model = models.Sequential()

    mod_conv = None
    #Load the VGG model

    if not mtype in ['vgg','vgg19','resnet','resnet101','inception','xception','simple'] : return False


    if 'vgg' in mtype :
        
        if mtype == 'vgg' :
            # Transfer Learning using VGG16
            mod_conv = VGG16(weights='imagenet', include_top=False, input_shape=(xsize, ysize, channels))
        if mtype == 'vgg19' :
            # Transfer Learning using VGG19
            mod_conv = VGG19(weights='imagenet', include_top=False, input_shape=(xsize, ysize, channels))

        # Freeze the layers except the last 4 layers
        for layer in mod_conv.layers[:-4]:
            layer.trainable = False

        # Add the vgg convolutional base model
        model.add(mod_conv)


    if 'resnet' in mtype :

        if mtype == 'resnet' :
            # Transfer Learning using ResNet50
            mod_conv = ResNet50(weights='imagenet', include_top=False, input_shape=(xsize, ysize, channels))

        if mtype == 'resnet101' :
            # Transfer Learning using ResNet100
            mod_conv = ResNet101(weights='imagenet', include_top=False, input_shape=(xsize, ysize, channels))

        output = mod_conv.layers[-1].output
        mod_conv = models.Model(mod_conv.input, output=output)

        # BatchNormalization Layers need to be trainable
        for layer in mod_conv.layers:
            if isinstance(layer, BatchNormalization):
                layer.trainable = True
            else:
                layer.trainable = False

        for layer in mod_conv.layers:
            if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97'] :
                layer.trainable = True

        # Add the resnet convolutional base model
        model.add(mod_conv)

    if mtype == 'inception' :

        # Transfer Learning using Inception V3
        mod_conv = InceptionV3(weights='imagenet', include_top=False, input_shape=(xsize, ysize, channels), classes=nlabels)

        # Freeze the layers except the last 4 layers
        for layer in mod_conv.layers:
            layer.trainable = False

        # Add the inception convolutional base model
        model.add(mod_conv)

    if mtype == 'xception' :
        # Transfer Learning using Xception
        mod_conv = Xception(weights='imagenet', include_top=False, input_shape=(xsize, ysize, channels))

        # Freeze the layers except the last 4 layers
        for layer in mod_conv.layers:
            layer.trainable = False

        # Add the inception convolutional base model
        model.add(mod_conv)


    if mtype == 'simple' :
        # Create Custom made CNN
        build_cnn(model=model, bn=bn, dropout=dropout)        
        

    # fully connected layers after 3D to 1D flat
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu', kernel_initializer = 'he_uniform'))
    model.add(layers.Dropout(dropout)) # <- comprobar si es necesario

    model.add(layers.Dense(nlabels, activation='softmax'))

    model.summary()
    
    model.compile(optimizer=optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model    


# ====================================================================
#  Build and set up model
# ====================================================================
def build_cnn(model, xsize=224,ysize=224,channels=3,nlabels=2, bn=False, dropout=0.4):

    # input of shape: (image_height, image_weight, image_channels)

    use_bias = True
    if bn :
        use_bias = False

    # convolution layers
    model.add(layers.Conv2D(32, (5,5), use_bias=use_bias, input_shape=(xsize,ysize,channels)))
    if bn :
        model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2,2)))
    #model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(64, (7,7), use_bias=use_bias, kernel_initializer = 'he_uniform'))
    if not bn :
        model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2,2)))
    #model.add(layers.Dropout(dropout))

    model.add(layers.Conv2D(64, (7,7), use_bias=use_bias, kernel_initializer = 'he_uniform'))
    if not bn :
        model.add(layers.BatchNormalization(momentum=0.9))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2,2)))    
    #model.add(layers.Dropout(dropout))


# ====================================================================
#  main
# ====================================================================
def main():

    # parse options from command line
    from optparse import OptionParser
    parser = OptionParser(usage = "usage: %prog arguments", version="%prog")
    parser.add_option("-r", "--report",      dest="report",  action="store_true",    help="Create report (default: %default)")
    parser.add_option("-n", "--bn",          dest="bn",      action="store_true",    help="Use Batch Normalization (default: %default)")
    parser.add_option("-m", "--model",       dest="model",                           help="Model to use (default: %default)")
    parser.add_option("-w", "--weights",     dest="weights",                         help="Load pre-made weights (default: %default)")
    parser.add_option("-d", "--dropout",     dest="dropout",                         help="Dropout level (default: %default)")
    parser.add_option("-l", "--lr",          dest="learning",                        help="Set learning rate (default: %default)")
    parser.add_option("-b", "--batch",       dest="batch",                           help="Set batch per epoch (default: %default)")
    parser.add_option("-e", "--epochs",      dest="epochs",                          help="Number of epochs to run (default: %default)")
    parser.add_option("-v", "--verbose",     dest="verbose",  action="store_true",   help="Print more information (default: %default)")
    parser.add_option("-t", "--test",        dest="test",     action="store_true",   help="Test weights and model (default: %default)")
    parser.set_defaults(verbose=True, report=False, weights=None, learning=0.00001, epochs=10, batch=None, bn=False, dropout=0.4, model='vgg')
    (options,args) = parser.parse_args()
    # parser.print_help()

    global wsize, hsize, bsize 


    # Set log directory for tensorboard
    now = datetime.datetime.now()
    logs_dir = now.strftime('./logs/%d%m%H%M')

    
    if not os.path.exists('saves'):
        os.mkdirs('saves')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

    from keras import backend as K
    K.clear_session()
    
    # create a data generator
    data_train = ImageDataGenerator()
    data_val = ImageDataGenerator()
    data_test = ImageDataGenerator()

    if options.model == "resnet" :
        bsize = 32

    if options.model == "inception" :
        wsize = hsize = 299
        data_train = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)
        data_val = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)
        data_test = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_v3.preprocess_input)

    if options.model == "xception" :
        wsize = hsize = 299
        data_train = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)
        data_val = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)
        data_test = ImageDataGenerator(preprocessing_function=tf.keras.applications.xception.preprocess_input)

    if options.model == "simple" :
        data_train = ImageDataGenerator(rescale=1./255)
        data_val = ImageDataGenerator(rescale=1./255)
        data_test = ImageDataGenerator(rescale=1./255)
        

    classes = getClasses("./InputImages/train")
    print("Using classes :",classes)

    # Load images for training
    train_it = data_train.flow_from_directory(
        directory=r"./InputImages/train",
        target_size=(wsize, hsize),
        color_mode="rgb",
        batch_size=bsize,
        class_mode="categorical",
        shuffle=True,
        seed=32
    )

    # Load images for validation
    val_it = data_val.flow_from_directory(
        directory=r"./InputImages/val",
        target_size=(wsize, hsize),
        color_mode="rgb",
        batch_size=bsize, 
        class_mode="categorical",
        shuffle=False,
        seed=42
    )

    if not options.test :
        if train_it.class_indices != val_it.class_indices :
            print((train_it.class_indices),(val_it.class_indices))
            print(" Somehow I did not get the order right!! Exiting...")
            sys.exit()

    # Create CNN model 
    model = build_model(nlabels=len(train_it.class_indices.keys()), xsize=wsize, ysize=hsize, lr=float(options.learning), bn=options.bn, dropout=options.dropout, mtype=options.model)

    if not model :
        print(" Please, select a valid CNN model")
        sys.exit(0)

    # Set Steps for training and validation
    if options.batch :
        STEP_SIZE_TRAIN=int(options.batch)
        STEP_SIZE_VALID=val_it.n//val_it.batch_size
    else :
        STEP_SIZE_TRAIN=train_it.n//train_it.batch_size
        STEP_SIZE_VALID=val_it.n//val_it.batch_size
    
    # Load pre-trained weights if needed
    if options.weights :
        try :
            print("\n Loading weights : ",options.weights)  
            model.load_weights(options.weights)  
            print(" ... done  \n")
        except :
            print (" Error Loading weights")
            sys.exit()


    if not options.test :
        # Create objects for tensorboard, called by training
        tensorboard = TrainValTensorBoard(log_dir=logs_dir, mod_name=options.model, write_graph=True, update_freq=500, val=val_it)

        # Run the model 
        history = model.fit_generator(train_it, epochs = int(options.epochs), steps_per_epoch=STEP_SIZE_TRAIN, validation_steps=STEP_SIZE_VALID, validation_data=val_it, callbacks=[tensorboard], shuffle=True)
    
        # Save model training
        model_weights_path = now.strftime('saves/weights_'+options.model+'.%d%m%H%M')
        model.save_weights(model_weights_path+".h5")

        # serialize model to JSON
        model_json = model.to_json()
        with open(model_weights_path+".json", "w") as json_file:
            json_file.write(model_json)

    
    printWrong = True
    # Create report 
    if options.report or options.test :

        # Load images for validation
        test_it = data_test.flow_from_directory(
            directory=r"./InputImages/test",
            target_size=(wsize, hsize),
            color_mode="rgb",
            batch_size=bsize, 
            class_mode="categorical",
            shuffle=False,
            seed=42
        )

        target_names = sorted(test_it.class_indices, key=test_it.class_indices.get)

        if train_it.class_indices != test_it.class_indices :        
            print(" Somehow I did not get the order right!! Exiting...")
            sys.exit()


        print(" Evaluating Predictions \n")
        Y_pred = model.predict_generator(test_it)#, val_it.n//val_it.batch_size )
        y_pred = np.argmax(Y_pred, axis=1)

        from sklearn.metrics import confusion_matrix
        from tensorboard_utils import plot_confusion_matrix
        
        print(" Computing Confusion Matrix \n")
        cnf_mat = confusion_matrix(test_it.classes, y_pred)

        print(cnf_mat)

        conf_matrix_norm = plot_confusion_matrix(cm=cnf_mat,classes=target_names,
                                                 normalize=True, tensor_name='Confusion Matrix Normalized',saveImg=True)
        
        print(classification_report(test_it.classes, y_pred, target_names=target_names))
        
        if printWrong :
            i = 0

            model_wrong_path = now.strftime('logs/wrong_'+options.model+'.%d%m%H%M')
            wrong_file  = open(model_wrong_path+".txt", "w") 

            for x in range(0,len(y_pred)) :

                if test_it.classes[x] != y_pred[x] :
                    real = target_names[test_it.classes[x]]
                    pred = target_names[y_pred[x]]
                    wrong = "File : "+str(test_it.filenames[x])+" -- "+real+"["+str(test_it.classes[x])+"] "+pred+"["+str(y_pred[x])+"]  -- "+str(Y_pred[x])+"\n"
                    wrong_file.write(wrong)
                    if i % 1000 == 0 :
                        print("Example ",wrong)
                    i += 1
                
            print(" Full List Saved at :",model_wrong_path+".txt")
            wrong_file.close()
# ====================================================================
#  __main__
# ====================================================================
if __name__ == '__main__':
    main()
