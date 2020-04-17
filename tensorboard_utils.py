#!/usr/bin/env python
# coding: utf-8

import os, sys, io, datetime
import numpy as np

from keras.callbacks import TensorBoard
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

import matplotlib
import matplotlib.pyplot as plt
import tfplot

def plot_roc_curve(false_pos, true_pos, auc, required_class) :

    fig = plt.figure(figsize=(10,10))  
    plt.plot(false_pos, true_pos, label='ROC (area = {:.3f})'.format(auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for {}'.format(required_class))
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc='best')
    plt.grid(True)   

    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag="ROC Curve "+required_class)
    return summary


def plot_confusion_matrix(cm, classes, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False, saveImg=False):
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
    if normalize:        
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    np.set_printoptions(precision=2)
    ###fig, ax = matplotlib.figure.Figure()

    fig = matplotlib.figure.Figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    if saveImg :
        fig = plt.figure(figsize=(7, 7), dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')
    
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
    if saveImg :
        fig.set_canvas(plt.gcf().canvas)
        fig.savefig('confusion_matrix.png')   # save the figure to file
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


class TrainValTensorBoard(TensorBoard):
    def __init__(self, mod_name="simple",log_dir='./logs', val=None, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
        self.val = val
        self.name = mod_name
        self._samples_seen = 0
        self._samples_seen_at_last_write = 0
        self._current_batch = 0
        self._total_batches_seen = 0
        self._total_val_batches_seen = 0

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics


        now = datetime.datetime.now()
        
        if (epoch + 1) % 10 == 0 :
            # Save model training
            model_weights_path = now.strftime('saves/weights_'+self.name+'.%d%m%H%M_'+str(epoch))
            self.model.save_weights(model_weights_path+".h5")

        # Add Confusion Matrix to Summaries
        self.val.reset()
        class_labels = list(self.val.class_indices.keys())
        print(class_labels,epoch)

        pred = self.model.predict_generator(self.val)#, self.val.n // self.val.batch_size + 1)
        max_pred = np.argmax(pred, axis=1)
        cnf_mat = confusion_matrix(self.val.classes, max_pred)


        conf_matrix_norm = plot_confusion_matrix(cm=cnf_mat,classes=class_labels,
                                                 normalize=True, tensor_name='Confusion Matrix Normalized')
        self.val_writer.add_summary(conf_matrix_norm, epoch) 
        
        conf_matrix = plot_confusion_matrix(cm=cnf_mat,classes=class_labels,
                                            normalize=False, tensor_name='Confusion Matrix')
        self.val_writer.add_summary(conf_matrix, epoch) 

        # Add ROC Curves
        label = 'ttbar'
        class_idx  = self.val.class_indices[label]
        pred_prob = pred[:, class_idx]  # prob of that particular class
        fpr, tpr, thresholds = roc_curve(self.val.classes,pred_prob, pos_label=class_idx)
        auc = roc_auc_score(self.val.classes,pred_prob)
        roc_plot = plot_roc_curve(fpr, tpr, auc, label) 
        self.val_writer.add_summary(roc_plot, epoch) 

        plt.close('all')

        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value#.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        plt.close('all')

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

    '''
    def on_batch_end(self, batch, logs=None):
        """Writes scalar summaries for metrics on every training batch."""
        # Don't output batch_size and batch number as Tensorboard summaries
        logs = logs or {}   
        self._samples_seen += logs.get('size', 1)
        samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
        if self.update_freq != 'epoch' and samples_seen_since >= self.update_freq:
            print(val_logs.items())
            batch_logs = {('batch_' + k): v
                          for k, v in logs.items()
                          if k not in ['batch', 'size', 'num_steps']}
#            self._write_custom_summaries(self._total_batches_seen, batch_logs)
            self._samples_seen_at_last_write = self._samples_seen
        self._total_batches_seen += 1
'''
