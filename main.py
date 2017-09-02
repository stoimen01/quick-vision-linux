import os
import tensorflow as tf

from model import ConvNet
from utils import pp
from server import Server

flags = tf.app.flags
flags.DEFINE_integer("input_size", 256, "The size of the input images [256]")
flags.DEFINE_integer("target_size", 1024, "The size of image to use (will be center cropped) [1024]")

flags.DEFINE_integer("epoch", 3, "Epoch to train [3]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

flags.DEFINE_string("train_dir", "data/outlook/train", "Path to the directory with train images [data/outlook/train]")
flags.DEFINE_string("test_dir", "data/outlook/test", "Path to the directory with test images [data/outlook/test]")
flags.DEFINE_string("checkpoint_dir", "checkpoints/outlook", "Directory name to save or load the checkpoint [checkpoint]")
flags.DEFINE_string("samples_dir", "samples", "Directory name to save the image samples [samples]")

flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_test", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_serve", False, "True for training, False for testing [False]")

flags.DEFINE_integer("port", 10000, "Server port [10 000]")

FLAGS = flags.FLAGS


def main(_):

    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.samples_dir):
        os.makedirs(FLAGS.samples_dir)

    if FLAGS.is_train or FLAGS.is_test:

        with tf.Session() as sess:

            conv_net = ConvNet(sess,
                               FLAGS.input_size,
                               FLAGS.target_size)

            if FLAGS.is_train:
                conv_net.train(FLAGS.learning_rate,
                               FLAGS.beta1,
                               FLAGS.epoch,
                               FLAGS.train_dir,
                               FLAGS.checkpoint_dir)

            elif FLAGS.is_test:
                conv_net.run_test(FLAGS.checkpoint_dir,
                                  FLAGS.test_dir,
                                  FLAGS.samples_dir)

    elif FLAGS.is_serve:

        server = Server(FLAGS.port,
                        FLAGS.input_size,
                        FLAGS.target_size,
                        FLAGS.checkpoint_dir)
        server.start()

    else:

        print("closing program. nothing to do")

if __name__ == '__main__':
    tf.app.run()
