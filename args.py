import tensorflow as tf


FLAGS = tf.app.flags.FLAGS # it might be same to use args

tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)#
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "5", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('log_dir', "/root/yq/SegNet/Logs", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "/root/yq/SegNet/CamVid/train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "/root/yq/SegNet/CamVid/test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "/root/yq/SegNet/CamVid/val.txt", """ path to CamVid val image """)
tf.app.flags.DEFINE_integer('max_steps', "20000", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "360", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "480", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_class', "11", """ total class number """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)
