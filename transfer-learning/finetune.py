import os, sys
import numpy as np
import tensorflow as tf
import datetime
from model import ResNetModel
sys.path.insert(0, '../utils')
from preprocessor import BatchPreprocessor

from sklearn.metrics import confusion_matrix
import tfplot
import matplotlib

tf.app.flags.DEFINE_float('learning_rate', 0.00001, 'Learning rate for adam optimizer')
tf.app.flags.DEFINE_integer('resnet_depth', 50, 'ResNet architecture to be used: 50, 101 or 152')
tf.app.flags.DEFINE_integer('num_epochs', 50, 'Number of epochs for training')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes')
tf.app.flags.DEFINE_integer('batch_size', 50, 'Batch size')
tf.app.flags.DEFINE_string('train_layers', 'fc', 'Finetuning layers, seperated by commas')
tf.app.flags.DEFINE_string('multi_scale', '', 'As preprocessing; scale the image randomly between 2 numbers and crop randomly at network\'s input size')
tf.app.flags.DEFINE_string('training_file', '../data/train.csv', 'Training dataset file')
tf.app.flags.DEFINE_string('val_file', '../data/val.csv', 'Validation dataset file')
tf.app.flags.DEFINE_string('tensorboard_root_dir', '../training', 'Root directory to put the training logs and weights')
tf.app.flags.DEFINE_integer('log_step', 10, 'Logging period in terms of iteration')
tf.app.flags.DEFINE_string('labels', 'ttbar,Wjets', 'Ordered list of labels separated by commas')

FLAGS = tf.app.flags.FLAGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main(_):
    # Create training directories
    now = datetime.datetime.now()
    train_dir_name = now.strftime('resnet_%Y%m%d_%H%M%S')
    train_dir = os.path.join(FLAGS.tensorboard_root_dir, train_dir_name)
    checkpoint_dir = os.path.join(train_dir, 'checkpoint')
    tensorboard_dir = os.path.join(train_dir, 'tensorboard')
    tensorboard_train_dir = os.path.join(tensorboard_dir, 'train')
    tensorboard_val_dir = os.path.join(tensorboard_dir, 'val')

    if not os.path.isdir(FLAGS.tensorboard_root_dir): os.mkdir(FLAGS.tensorboard_root_dir)
    if not os.path.isdir(train_dir): os.mkdir(train_dir)
    if not os.path.isdir(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.isdir(tensorboard_dir): os.mkdir(tensorboard_dir)
    if not os.path.isdir(tensorboard_train_dir): os.mkdir(tensorboard_train_dir)
    if not os.path.isdir(tensorboard_val_dir): os.mkdir(tensorboard_val_dir)

    # Write flags to txt
    flags_file_path = os.path.join(train_dir, 'flags.txt')
    flags_file = open(flags_file_path, 'w')
    flags_file.write('learning_rate={}\n'.format(FLAGS.learning_rate))
    flags_file.write('resnet_depth={}\n'.format(FLAGS.resnet_depth))
    flags_file.write('num_epochs={}\n'.format(FLAGS.num_epochs))
    flags_file.write('batch_size={}\n'.format(FLAGS.batch_size))
    flags_file.write('train_layers={}\n'.format(FLAGS.train_layers))
    flags_file.write('multi_scale={}\n'.format(FLAGS.multi_scale))
    flags_file.write('tensorboard_root_dir={}\n'.format(FLAGS.tensorboard_root_dir))
    flags_file.write('log_step={}\n'.format(FLAGS.log_step))
    flags_file.close()

    # Placeholders
    x = tf.placeholder(tf.float32, [FLAGS.batch_size, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, FLAGS.num_classes])
    is_training = tf.placeholder('bool', [])

    # Model
    train_layers = FLAGS.train_layers.split(',')
    model = ResNetModel(is_training, depth=FLAGS.resnet_depth, num_classes=FLAGS.num_classes)
    loss = model.loss(x, y)
    train_op = model.optimize(FLAGS.learning_rate, train_layers)

    # Training accuracy of the model
    corr_pred = tf.argmax(model.prob, 1)
    correct_pred = tf.equal(tf.argmax(model.prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Summaries
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', accuracy)
    merged_summary = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(tensorboard_train_dir)
    val_writer = tf.summary.FileWriter(tensorboard_val_dir)
    saver = tf.train.Saver()

    # Batch preprocessors
    multi_scale = FLAGS.multi_scale.split(',')
    if len(multi_scale) == 2:
        multi_scale = [int(multi_scale[0]), int(multi_scale[1])]
    else:
        multi_scale = None

    train_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.training_file, num_classes=FLAGS.num_classes,
                                           output_size=[224, 224], horizontal_flip=True, shuffle=True, multi_scale=multi_scale)
    val_preprocessor = BatchPreprocessor(dataset_file_path=FLAGS.val_file, num_classes=FLAGS.num_classes, output_size=[224, 224])

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = np.floor(len(train_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)
    val_batches_per_epoch = np.floor(len(val_preprocessor.labels) / FLAGS.batch_size).astype(np.int16)


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer.add_graph(sess.graph)

        # Load the pretrained weights
        model.load_original_weights(sess, skip_layers=train_layers)

        # Directly restore (your model should be exactly the same with checkpoint)
        # saver.restore(sess, "/Users/dgurkaynak/Projects/marvel-training/alexnet64-fc6/model_epoch10.ckpt")

        print(" {} | Start training...".format(datetime.datetime.now().strftime("%H:%M:%S")))
        print(" {} | Open Tensorboard at --logdir {}".format(datetime.datetime.now().strftime("%H:%M:%S"), tensorboard_dir))


        #train_batches_per_epoch = 11
        #val_batches_per_epoch = 2

        for epoch in range(FLAGS.num_epochs):
            print(" {} | Epoch number: {}".format(datetime.datetime.now().strftime("%H:%M:%S"), epoch+1))
            step = 1

            # Start training
            while step < train_batches_per_epoch:
                batch_xs, batch_ys, ys = train_preprocessor.next_batch(FLAGS.batch_size)
                sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, is_training: True})

                # Logging
                if step % FLAGS.log_step == 0:
                    s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, is_training: False})
                    train_writer.add_summary(s, epoch * train_batches_per_epoch + step)
                    train_writer.flush() 
 
                    batch_tx, batch_ty, ty = val_preprocessor.next_batch(FLAGS.batch_size)
                    v = sess.run(merged_summary, feed_dict={x: batch_tx, y: batch_ty, is_training: False})
                    val_writer.add_summary(v, epoch * train_batches_per_epoch + step)
                    val_writer.flush() 

                step += 1

            # Reset the dataset pointers
            val_preprocessor.reset_pointer()

            # Epoch completed, start validation 
            print(" {} | Start validation".format(datetime.datetime.now().strftime("%H:%M:%S")))
            test_acc = 0.
            test_count = 0

            val_pred = []
            val_ty = []
            for _ in range(val_batches_per_epoch):
                batch_tx, batch_ty, ty = val_preprocessor.next_batch(FLAGS.batch_size)
                acc, pred = sess.run([accuracy, corr_pred], feed_dict={x: batch_tx, y: batch_ty, is_training: False})

                val_pred.extend(pred)
                val_ty.extend(ty)

                try :
                    confusion += tf.confusion_matrix(labels=ty, predictions=pred, num_classes=FLAGS.num_classes)
                except :
                    confusion = tf.confusion_matrix(labels=ty, predictions=pred, num_classes=FLAGS.num_classes)

                test_acc += acc
                test_count += 1
                
            test_acc /= test_count

            s = tf.Summary(value=[tf.Summary.Value(tag="val_accuracy", simple_value=test_acc)])
            val_writer.add_summary(s, epoch * train_batches_per_epoch + step - 1)

            print(" {} | Validation Accuracy = {:.4f}".format(datetime.datetime.now().strftime("%H:%M:%S"), test_acc))

            # Confusion Matrix
            with tf.Session():
                conf_out = tf.Tensor.eval(confusion, feed_dict=None, session=None)

            conf_matrix = plot_confusion_matrix(correct_labels=val_ty, 
                                                predict_labels=val_pred, labels=FLAGS.labels, 
                                                tensor_name='Confusion Matrix')
            val_writer.add_summary(conf_matrix, epoch+1)
            
            conf_matrix_norm = plot_confusion_matrix(correct_labels=val_ty, 
                                                     predict_labels=val_pred, labels=FLAGS.labels, 
                                                     normalize=True, tensor_name='Confusion Matrix Normalized')
            val_writer.add_summary(conf_matrix_norm, epoch+1)  

            # Reset the dataset pointers
            val_preprocessor.reset_pointer()
            train_preprocessor.reset_pointer()

            print(" {} | Saving checkpoint of model...".format(datetime.datetime.now().strftime("%H:%M:%S")))

            #save checkpoint of the model
            checkpoint_path = os.path.join(checkpoint_dir, 'model_epoch'+str(epoch+1)+'.ckpt')
            save_path = saver.save(sess, checkpoint_path)

            print(" {} | Model checkpoint saved at {}".format(datetime.datetime.now().strftime("%H:%M:%S"), checkpoint_path))

def plot_confusion_matrix(correct_labels, predict_labels, labels, title='Confusion matrix', tensor_name = 'MyFigure/image', normalize=False):
    ''' 
    Parameters:
    correct_labels                  : These are your true classification categories.
    predict_labels                  : These are you predicted classification categories
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
    
    cm = confusion_matrix(correct_labels, predict_labels, labels=num_labels)
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



if __name__ == '__main__':
    tf.app.run()
