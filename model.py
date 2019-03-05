from args import *

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
# modules
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
from datasets import ArtPrint
from torch.utils.data import DataLoader, ConcatDataset, random_split#, SequentialSampler #yike: add SequentialSampler
from os.path import join, basename, dirname
from Inputs import *


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = FLAGS.learning_rate#0.001      # Initial learning rate. yike
EVAL_BATCH_SIZE = FLAGS.batch_size
BATCH_SIZE = FLAGS.batch_size
# for CamVid
IMAGE_HEIGHT = FLAGS.image_h
IMAGE_WIDTH = FLAGS.image_w
IMAGE_DEPTH = FLAGS.image_c

NUM_CLASSES = FLAGS.num_class+1
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 367
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 101
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE





def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def loss(logits, labels):
  """
      loss func without re-weighting
  """
  # Calculate the average cross entropy loss across the batch.
  logits = tf.reshape(logits, (-1,NUM_CLASSES))
  labels = tf.reshape(labels, [-1])

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def weighted_loss(logits, labels, num_classes, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):
        #print('w_llll')
        logits = tf.reshape(logits, (-1, num_classes))
        #print(logits.get_shape())
        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))
#        print(label_flat.get_shape())

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
#        print(labels.get_shape())

        softmax = tf.nn.softmax(logits)
#        print(softmax.get_shape())
#        print(epsilon.get_shape())

#        print((labels * tf.log(softmax + epsilon)).get_shape())
#        print(head.shape)
#        print(tf.multiply(labels * tf.log(softmax + epsilon), head))
        
        cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])
#        print(cross_entropy.get_shape())

        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
#        print(cross_entropy_mean.get_shape())
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
#        print(loss)      

    return loss

def cal_loss(logits, labels): # yike: need to change
#    loss_weight = np.array([
#      0.2595,
#      0.1826,
#      4.5640,
#      0.1417,
#      0.9051,
#      0.3826,
#      9.6446,
#      1.8418,
#      0.6823,
#      6.2478,
#      7.3614,
#      1.0974]) # class 0~11
    loss_weight=np.array([0.4,0.6]) #yike

    labels = tf.cast(labels, tf.int32)
    print('clll')
    print(logits.get_shape())
    print(labels.get_shape())
    # return loss(logits, labels)
    return weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
      kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  # sess_temp = tf.InteractiveSession()
  sess_temp = tf.global_variables_initializer()
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def inference(images, labels, batch_size, phase_train):
    print('GGG')
    print(images.get_shape())
    # norm1
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                name='norm1')
    print(norm1.get_shape())
    # conv1
    conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], 64], phase_train, name="conv1") # yike: 7 too large? how about 3?
    print(conv1.get_shape())
    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    print(pool1.get_shape())
    print(pool1_indices.get_shape())
    # conv2
    conv2 = conv_layer_with_bn(pool1, [7, 7, 64, 64], phase_train, name="conv2")
    

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    print('22222')
    print(pool2.get_shape())
    print(pool2_indices.get_shape())


    # conv3
    conv3 = conv_layer_with_bn(pool2, [7, 7, 64, 64], phase_train, name="conv3")

    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    print('33333')
    print(pool3.get_shape())
    print(pool3_indices.get_shape())



    # conv4
    conv4 = conv_layer_with_bn(pool3, [7, 7, 64, 64], phase_train, name="conv4")

    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    print('44444')
    print(pool4.get_shape())
    print(pool4_indices.get_shape())

    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(pool4, [2, 2, 64, 64], [batch_size, 45, 60, 64], 2, "up4")
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, 64, 64], phase_train, False, name="conv_decode4")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3= deconv_layer(conv_decode4, [2, 2, 64, 64], [batch_size, 90, 120, 64], 2, "up3")
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, 64, 64], phase_train, False, name="conv_decode3")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2= deconv_layer(conv_decode3, [2, 2, 64, 64], [batch_size, 180, 240, 64], 2, "up2")
    # decode 2
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, 64, 64], phase_train, False, name="conv_decode2")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1=deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 64], 2, "up1") # yike !!!! deconv_layer(conv_decode2, [2, 2, 64, 64], [batch_size, 360, 480, 64], 2, "up1")
    # decode4
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, 64, 64], phase_train, False, name="conv_decode1")
    print('d111111')
    print(conv_decode1.get_shape())
    
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, 64, NUM_CLASSES],
                                           initializer=msra_initializer(1, 64),
                                           wd=0.0005)
      conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
      print('cv')
      print(conv.get_shape())
      biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
      print(biases.get_shape())
      conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
      print(conv_classifier.get_shape())
    logit = conv_classifier
    print('LLL')
    print(labels)
    print(conv_classifier)
    
    loss = cal_loss(conv_classifier, labels)

    return loss, logit

def train(total_loss, global_step):
    #total_sample = 274 yike: ok to comment out?
    #num_batches_per_epoch = 274/1 yike: ok to comment out?
    """ fix lr """
    print(total_loss)
    lr = INITIAL_LEARNING_RATE
    loss_averages_op = _add_loss_summaries(total_loss)
    print(loss_averages_op)
    print('11111')
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      #print('try...')
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
      #print(grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op

def test(FLAGS):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  train_dir = FLAGS.log_dir # /tmp3/first350/TensorFlow/Logs
  test_dir = FLAGS.test_dir # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
  test_ckpt = FLAGS.testing
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  # testing should set BATCH_SIZE = 1
  batch_size = 1

  image_filenames, label_filenames = get_filename_list(test_dir)

  test_data_node = tf.placeholder(
        tf.float32,
        shape=[batch_size, image_h, image_w, image_c])

  test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, 360, 480, 1])

  phase_train = tf.placeholder(tf.bool, name='phase_train')

  loss, logits = inference(test_data_node, test_labels_node, batch_size, phase_train)

  pred = tf.argmax(logits, axis=3)
  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(
                      MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore)

  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)

  with tf.Session() as sess:
    # Load checkpoint
    saver.restore(sess, test_ckpt )

    images, labels = get_all_test_data(image_filenames, label_filenames)

    threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for image_batch, label_batch  in zip(images, labels):

      feed_dict = {
        test_data_node: image_batch,
        test_labels_node: label_batch,
        phase_train: False
      }

      dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
      # output_image to verify
      if (FLAGS.save_image):
          writeImage(im[0], 'testing_image.png')
          # writeImage(im[0], 'out_image/'+str(image_filenames[count]).split('/')[-1])

      hist += get_hist(dense_prediction, label_batch)
      # count+=1
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))

def training(FLAGS, loader=None,validateloader=None,testloader=None, is_finetune=False):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  train_dir = FLAGS.log_dir # /tmp3/first350/TensorFlow/Logs
  image_dir = FLAGS.image_dir # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
  val_dir = FLAGS.val_dir # /tmp3/first350/SegNet-Tutorial/CamVid/val.txt
  finetune_ckpt = FLAGS.finetune
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  # should be changed if your model stored by different convention
  startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])
######################################################################
#  image_filenames, label_filenames = get_filename_list(image_dir)
#  val_image_filenames, val_label_filenames = get_filename_list(val_dir)
#  image_filenames, label_filenames, val_image_filenames, val_label_filenames=get_filename_list_train_val(image_dir,train_per=0.9) #yike
#  print('img_fnames')
#  print(type(image_filenames))
  #print(len(image_filenames))
  #print(image_filenames[10])
  print(len(validateloader))
  print('+++')
  with tf.Graph().as_default():

    train_data_node = tf.placeholder( tf.float32, shape=[None, image_h, image_w, image_c]) #batch_size

    train_labels_node = tf.placeholder(tf.int64, shape=[None, image_h, image_w, 1]) #batch_size

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    global_step = tf.Variable(0, trainable=False)

    # For CamVid
########################################################
    #images, labels = ArtPrintVidInputs(image_filenames, label_filenames, batch_size)
    # REPLACE WITH DATALOADER images,labels=CamVidInputs(image_filenames, label_filenames, batch_size)

#    print('aaa')
#    print(images.shape)
#    print(images[0])
#########################################################
    #val_images, val_labels = ArtPrintVidInputs(val_image_filenames, val_label_filenames, batch_size)
    # REPLACE WITH DATALOADER val_images, val_labels=CamVidInputs(val_image_filenames, val_label_filenames, batch_size)
#    print('ttt')
#    print(val_images.shape)
#    print(train_data_node)
#    print(train_labels_node)
    # Build a Graph that computes the logits predictions from the inference model.
    loss, eval_prediction = inference(train_data_node, train_labels_node, batch_size, phase_train)
#    print('le')
#    print(loss)
#    print(eval_prediction)
    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = train(loss, global_step)
#    print('ffffff')
    saver = tf.train.Saver(tf.global_variables())

#    summary_op = tf.summary.merge_all()

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
      # Build an initialization operation to run below.
      if (is_finetune == True):
          saver.restore(sess, finetune_ckpt )
      else:
          init = tf.global_variables_initializer()
          sess.run(init)

#NOT NEEDED SINCE WE ARE USING DATALOADER
      # Start the queue runners.
#      coord = tf.train.Coordinator()
#      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#NOT NEEDED SINCE WE USE COMET TO LOG OUR RESULTS
      # Summery placeholders
#      summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
#      average_pl = tf.placeholder(tf.float32)
#      acc_pl = tf.placeholder(tf.float32)
#      iu_pl = tf.placeholder(tf.float32)
#      average_summary = tf.summary.scalar("test_average_loss", average_pl)
#      acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
#      iu_summary = tf.summary.scalar("Mean_IU", iu_pl)

      # for step in range(startstep, startstep + max_steps):
      epoch = 0  # number of training epochs since start
      bestAcc = float('-inf')  # best valdiation accuracy rate
      while True:
        epoch+=1
        for _step, (image_batch, label_batch) in enumerate(loader):
          step=startstep+_step
          #print('have a try')
          # GET RID OF THIS!!!!!!!!!!!!!!!!!1 image_batch ,label_batch = sess.run([images, labels])
          # print('have a try')
          image_batch=image_batch.numpy()
          label_batch=label_batch.numpy()
          #print(label_batch.get_shape())
          # since we still use mini-batches in validation, still set bn-layer phase_train = True
          #image_batch, label_batch = YIKEDATALOADER.__next__()
          feed_dict = {
            train_data_node: image_batch,
            train_labels_node: label_batch,
            phase_train: True
          }
          start_time = time.time()
          _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
          duration = time.time() - start_time

          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

          if step % 10 == 0:
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

          # eval current training batch pre-class accuracy
            pred = sess.run(eval_prediction, feed_dict=feed_dict)
            print(pred.shape)
            print('~~~')
            #print(sum(pred>0.5))
            #print(sum(pred>0.5)/sum(pred>-10000))
            per_class_acc(pred, label_batch)

          if _step % 500 == 0 or _step+1==len(image_batch):
            print("start validating.....")
            total_val_loss = 0.0
            hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
#            for test_step in range(int(TEST_ITER)):
            print(len(validateloader))
            for test_step, (val_images_batch,  val_labels_batch) in enumerate(validateloader):# val_labels_batch = sess.run([val_images, val_labels])
              val_images_batch=val_images_batch.numpy()
              val_labels_batch=val_labels_batch.numpy()
              _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
                train_data_node: val_images_batch,
                train_labels_node: val_labels_batch,
                phase_train: True
              })
              total_val_loss += _val_loss
              hist += get_hist(_val_pred, val_labels_batch)
            
            val_loss=total_val_loss / len(validateloader)*batch_size
            acc_total = np.diag(hist).sum() / hist.sum()
            print('validation/total accuracy: '+str(acc_total))
            acc_class_total=[]
            time_anchor=(_step+FLAGS.max_epoch*(epoch-1))/(FLAGS.max_epoch*len(loader))
            for ii in range(NUM_CLASSES):
              if float(hist.sum(1)[ii]) == 0:
                acc = 0.0
              else:
                acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
              acc_class_total.append(acc)
              
              experiment.log_metric('validation/loss', val_loss, time_anchor)
            #print("    class # %d accuracy = %f "%(ii,acc))
            iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
            mean_iu=np.nanmean(iu)
            
            print('percentage: ' + str(time_anchor))
            print('validation/loss: '+str(val_loss))
            print('validation/total accuracy: '+str(acc_total))
            print('validation/class_0 accuracy: '+str(acc_class_total[0]))
            print('validation/class_1 accuracy: '+str(acc_class_total[1]))
            
            experiment.log_metric('validation/loss', val_loss, time_anchor)
            experiment.log_metric('validation/total accuracy', acc_total, time_anchor)
            experiment.log_metric('validation/class_0 accuracy',acc_class_total[0], time_anchor)
            experiment.log_metric('validation/class_1 accuracy',acc_class_total[1], time_anchor)

        if acc_total>bestAcc:
          print('Total accuracy rate improved')
          checkpoint_path=os.path.join(train_dir+'/'+FLAGS.name+'/'+'model.ckpt')
          saver.save(sess,checkpoint_path,global_step=epoch) 
          bestAcc=acc_total
          noImprovementSince=0
          open(checkpoint_path.replace('model.ckpt','accuracy.txt'), 'w').write(
        'Validation total class accuracy rate of saved model: %f%%' % (bestAcc * 100.0))
          experiment.log_metric('best/total accuracy', acc_total, time_anchor)
          experiment.log_metric('best/class 0 accuracy', acc_class_total[0], time_anchor)
          experiment.log_metric('best/class 1 accuracy', acc_class_total[1], time_anchor)
        else:
          print('Total accuracy rate not improved')
          noImprovementSince+1
        
        if epoch>=FLAGS.max_epoch: print('Done with training at epoch', epoch, 'sigoptObservation='+str(bestAcc)); break
'''        
      for im_show_batch, label_show_batch in validateloader:
        feed_dict = {
            train_data_node: image_show_batch,
            train_labels_node: label_show_batch,
            phase_train: True
          }
        pred = sess.run(eval_prediction, feed_dict=feed_dict)
        result=pred.argmax(3)
        for idx in range(batch_size):
          colored=cv2.cvtColor(np.squeeze(np.image_show_batch[idx]),cv2.COLOR_GRAY2BGR)
          break
'''
#            test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
#            acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
#            iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
#            print_hist_summery(hist)
#            print(" end validating.... ")

#            summary_str = sess.run(summary_op, feed_dict=feed_dict)
#            summary_writer.add_summary(summary_str, step)
#            summary_writer.add_summary(test_summary_str, step)
#            summary_writer.add_summary(acc_summary_str, step)
#            summary_writer.add_summary(iu_summary_str, step)
      #     Save the model checkpoint periodically.
