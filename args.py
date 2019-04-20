#from comet_ml import Experiment
#experiment = Experiment(api_key="YkPEmantOag1R1VOJmXz11hmt", parse_args=False, project_name='misc')
from os.path import join, basename, dirname
import tensorflow as tf
import shutil
import os
import sys
#import argparse

home = os.environ['HOME']

FLAGS = tf.app.flags.FLAGS 

# system basics
tf.app.flags.DEFINE_string('name', "debug", """ experiment name """)
tf.app.flags.DEFINE_integer('gpu', "0", """ gpu numbers """)

# image parameters 
tf.app.flags.DEFINE_integer('image_h', "32", """ image height """) #('image_h', "360", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "128", """ image width """)#('image_w', "480", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "1", """ image channel (Grey) """)#('image_c', "3", """ image channel (RGB) """)

# training hyperparam
tf.app.flags.DEFINE_integer('batch_size', "10", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_integer('max_epoch','2',"""max epoch numbers""")

# file paths
tf.app.flags.DEFINE_string('ckpt_root', "/home/classify_copy/misc_test/ckpt", """ dir to store ckpt """) # log_dir !!!!!
tf.app.flags.DEFINE_string('data_root', "/home/classify_copy/misc_test/datasets", """ root to any data folder """)
'''
print(FLAGS.name)
tf.app.flags.DEFINE_string('ckptpath', "", """ dir to store ckpt """)
ckptroot = FLAGS.ckpt_root
FLAGS.ckptpath = join(ckptroot, FLAGS.name)
if FLAGS.name=='debug': shutil.rmtree(FLAGS.ckptpath, ignore_errors=True)
os.makedirs(FLAGS.ckptpath, exist_ok=True)
'''

#tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)#
#tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)

#tf.app.flags.DEFINE_string('image_dir', "/root/yq/SegNet/CamVid/train.txt", """ path to CamVid image """) # still useful
#tf.app.flags.DEFINE_string('image_dir', "/root/datasets/artifact_images/databook.txt", """ path to training images folder """) # still useful

###tf.app.flags.DEFINE_string('test_dir', "/root/yq/SegNet/CamVid/test.txt", """ path to CamVid test image """) # maybe

###tf.app.flags.DEFINE_string('val_dir', "/root/yq/SegNet/CamVid/val.txt", """ path to CamVid val image """) # no more useful
#tf.app.flags.DEFINE_integer('max_steps', "20000", """ max_steps """) # no more useful

 #('num_class', "11", """ total class number """)
#tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)

#tf.app.flags.DEFINE_string('ckptpath', "", """ dir to store ckpt """)


