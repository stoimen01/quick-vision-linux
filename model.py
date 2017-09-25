from __future__ import division
import time
from glob import glob
import os
from ops import *
from six.moves import xrange
from subpixel import PS
from utils import *


class ConvNet(object):

    def __init__(self, sess, input_size, target_size, y_dim=None, z_dim=100, gf_dim=64, color_dim=3):

        self.sess = sess

        self.input_size = input_size
        self.input_shape = [1, input_size, input_size, color_dim]

        self.target_size = target_size
        self.target_shape = [1, target_size, target_size, color_dim]

        self.color_dim = color_dim

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim

        # Building the model #

        self.inputs = tf.placeholder(tf.float32, self.input_shape, name='real_images')

        self.images = tf.placeholder(tf.float32, self.target_shape, name='real_images')

        try:
            self.up_inputs = tf.image.resize_images(self.inputs, self.target_size, self.target_size,
                                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        except ValueError:

            self.up_inputs = tf.image.resize_images(self.inputs, [self.target_size, self.target_size],
                                                    tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        self.G = self.create_network(self.inputs)

        self.G_sum = tf.summary.image("G", self.G)

        self.g_loss = tf.reduce_mean(tf.square(self.images - self.G))

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def create_network(self, z):

        self.h0, self.h0_w, self.h0_b = deconv2d(z, [1, self.input_size, self.input_size, self.gf_dim],
                                                 k_h=1, k_w=1, d_h=1, d_w=1, name='g_h0', with_w=True)
        h0 = lrelu(self.h0)

        self.h1, self.h1_w, self.h1_b = deconv2d(h0, [1, self.input_size, self.input_size, self.gf_dim],
                                                 name='g_h1', d_h=1, d_w=1, with_w=True)
        h1 = lrelu(self.h1)

        h2, self.h2_w, self.h2_b = deconv2d(h1, [1, self.input_size, self.input_size, 3*16],
                                            d_h=1, d_w=1, name='g_h2', with_w=True)
        h2 = PS(h2, 4, color=True)

        return tf.nn.tanh(h2)

    def train(self, learning_rate, beta1, epoch, train_dir, c_dir):

        network_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        self.g_sum = tf.summary.merge([self.G_sum, self.g_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(c_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        in_img = np.zeros((1, self.input_size, self.input_size, self.color_dim)).astype(np.float32)
        target_img = np.zeros((1, self.target_size, self.target_size, self.color_dim)).astype(np.float32)


        for epoch in xrange(epoch):

            for file in sorted(glob(os.path.join(train_dir, "*.jpg"))):

                img_transformed = transform_img(crop_img(file, self.target_size))
                image_resized = resize(img_transformed, self.input_size)

                in_img[0] = image_resized
                target_img[0] = img_transformed

                t1 = time.time()
                _, summary_str, err_g = self.sess.run([network_optimizer, self.g_sum, self.g_loss],
                    feed_dict = { self.inputs: in_img, self.images: target_img })
                self.writer.add_summary(summary_str, counter)

                print("Epoch: [%2d]  time: %4.4f, g_loss: %.8f, run time: %4.4f" \
                      % (epoch, time.time() - start_time, err_g, time.time() - t1))

                if np.mod(counter, 300) == 2:
                    self.save(c_dir, counter)
                counter += 1

        print("training finished in " + str(time.time() - start_time) + " seconds")

    def run_test(self, c_dir, test_dir, samples_dir):

        if not self.load(c_dir):
            print("Unable to load checkpoint")
            return

        data = sorted(glob(os.path.join(test_dir, "*.jpg")))
        in_arr = np.zeros((1, self.input_size, self.input_size, self.color_dim)).astype(np.float32)
        target_arr = np.zeros((1, self.target_size, self.target_size, self.color_dim)).astype(np.float32)

        counter = 1
        for file in data:

            target_img = transform_img(crop_img(file, self.target_size))
            target_arr[:] = target_img

            input_img = resize(target_img, self.input_size)
            in_arr[:] = input_img

            t1 = time.time()
            sample, g_loss = self.sess.run(
                [self.G, self.g_loss],
                feed_dict={self.inputs: in_arr, self.images: target_arr}
            )
            print("RPS : " + str(1 / (time.time() - t1)))

            save_image(in_arr, './' + samples_dir + '/input_%s.png' % counter)
            save_image(sample, './' + samples_dir + '/sample_%s.png' % counter)
            save_image(target_arr, './' + samples_dir + '/target_%s.png' % counter)

            counter += 1

    def pass_forward(self, img):

        in_img = resize(transform_img(img), self.input_size)
        in_arr = np.zeros((1, self.input_size, self.input_size, self.color_dim)).astype(np.float32)
        in_arr[:] = in_img

        sample_arr = np.zeros((1, self.target_size, self.target_size, self.color_dim)).astype(np.float32)

        samples, g_loss = self.sess.run(
            [self.G, self.g_loss],
            feed_dict={self.inputs: in_arr, self.images: sample_arr})

        res = inverse_transform(samples)
        return res[0]

    def save(self, checkpoint_dir, step):
        model_name = "conv.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, c_dir):
        ckpt = tf.train.get_checkpoint_state(c_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(c_dir, ckpt_name))
            return True
        else:
            return False
