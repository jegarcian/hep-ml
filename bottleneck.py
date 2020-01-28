'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import os, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras import optimizers
from keras import utils as np_utils
import tensorflow as tf
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
import tfplot
import matplotlib

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = '/home/jgarcian/HEPML/images/train'
validation_data_dir = '/home/jgarcian/HEPML/images/val'
nb_train_samples = 51082
nb_validation_samples = 5676
epochs = 50
batch_size = 16
num_classes = 2
classes = "ttbar,Wjets"

def plot_confusion_matrix(cm, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    ''' 
    Parameters:
    labels                          : This is a lit of labels which will be used to display the axix labels
    title='Confusion matrix'        : Title for your matrix
    tensor_name = 'MyFigure/image'  : Name for the output summay tensor

    Returns:
    summary: TensorFlow summary 
    
    Other itema to note:
    - Depending on the number of category and the data , you may have to modify the figzie, font sizes etc. 
    - Currently, some of the ticks dont line up due to rotations.
    '''

    num_labels = []
    for x in range(0,len(labels.split(","))) :
      num_labels.append(x)
    
    if normalize:        
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')
    
    classes = labels.split(",")

    tick_marks = np.arange(len(classes))
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=12, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()
    
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    
    import itertools    
    fmt = '.2f' if normalize else 'd'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt) if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=20, verticalalignment='center', color= "black")
        
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', classes=None, val=None, val_truth=None, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
        self.val_truth = val_truth
        self.val = val
        self.classes = classes

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics

        pred = self.model.predict(self.val)
        max_pred = np.argmax(pred, axis=1)
        max_y = np.argmax(self.val_truth, axis=1)
        cnf_mat = confusion_matrix(max_y, max_pred)

        conf_matrix_norm = plot_confusion_matrix(cm=cnf_mat,labels=self.classes,
                                                 normalize=True, tensor_name='Confusion Matrix Normalized')
        self.val_writer.add_summary(conf_matrix_norm, epoch) 

        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

def save_bottlebeck_features():
    datagen = ImageDataGenerator()

    # build the VGG16 network
    model = applications.ResNet50(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, len(generator))

    np.save(open('bottleneck_features_train.npy', 'bw'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, len(generator))

    np.save(open('bottleneck_features_validation.npy', 'bw'),
            bottleneck_features_validation)

def train_top_model():
    
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    train_labels = np.array(
        [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))
    train_labels = np_utils.to_categorical(train_labels.tolist(), num_classes)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))
    validation_labels = np_utils.to_categorical(validation_labels.tolist(), num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(lr=0.00001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels),
                        callbacks=[TrainValTensorBoard(write_graph=True,update_freq=500,val_truth=validation_labels.tolist(),val=validation_data,classes=classes)])

    model.save_weights(top_model_weights_path)

#    pred = model.predict(validation_data)
#    print(pred)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

#save_bottlebeck_features()
train_top_model()
