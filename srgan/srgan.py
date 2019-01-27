"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from scipy.special import expit

import keras.backend as K
from keras.datasets import mnist
#from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, concatenate, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, Layer
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.initializers import VarianceScaling
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, History, TensorBoard
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

import keras.backend as K

from SpectralNormalizationKeras import DenseSN, ConvSN2D

import sys

import numpy as np


def discriminator_loss(y_true, y_pred):
    shape = K.shape(y_pred)
    d_real = y_pred[:shape[0]//2, :]
    d_fake = y_pred[shape[0]//2:, :]

    loss_real = K.mean(K.relu(-1 + d_real))
    loss_fake = K.mean(K.relu(1 + d_fake))
    return loss_real + loss_fake


def generator_loss(y_true, y_pred):
    return -K.mean(y_pred)


class SelfAttention(Layer):
    def __init__(self, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.

        input_dim = input_shape[-1]
        kernel_shape = (1, 1, input_dim, input_dim // 8)

        self.kernel_f = self.add_weight(name='kernel_f',
                                      shape=kernel_shape,
                                      initializer='uniform',
                                      trainable=True)

        self.kernel_g = self.add_weight(name='kernel_g',
                                      shape=kernel_shape,
                                      initializer='uniform',
                                      trainable=True)

        self.kernel_h = self.add_weight(name='kernel_h',
                                      shape=(1, 1, input_dim, input_dim),
                                      initializer='uniform',
                                      trainable=True)

        self._gamma = self.add_weight(name='scale',
                                     shape=(1,),
                                     initializer='zeros',
                                     trainable=True)

        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        _, h, w, c = K.int_shape(x)

        b = K.shape(x)[0]

        f = K.reshape(K.conv2d(x, self.kernel_f, padding='same'), (b, h*w, -1))
        g = K.permute_dimensions(K.reshape(K.conv2d(x, self.kernel_g, padding='same'), (b, h*w, -1)), (0, 2, 1))
        s = K.batch_dot(f, g)
        beta = K.softmax(s)

        h = K.reshape(K.conv2d(x, self.kernel_h, padding='same'),
                                           (b, h*w, c))

        out = K.batch_dot(beta, h)

        out = K.reshape(out, K.shape(x))

        out = self._gamma * out + x

        return out

    def compute_output_shape(self, input_shape):
        return input_shape

class SmallInitialization(VarianceScaling):
    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def __call__(self, shape, dtype=None):
        return self.scale * super().__call__(shape, dtype)


def rel_avg_loss(x_r, x_f):
    return K.sigmoid(x_r - K.mean(x_f))

def d_loss(y_real, y_pred):
    batch_size = K.shape(y_pred)[0]
    x_r = y_pred[:batch_size // 2, :]
    x_f = y_pred[batch_size // 2:, :]

    d_ra_real = rel_avg_loss(x_r, x_f)
    d_ra_fake = rel_avg_loss(x_f, x_r)
    y_pred = K.concatenate([d_ra_real, d_ra_fake], axis=0)

    return K.mean(K.binary_crossentropy(y_real, y_pred), axis=-1)

def g_loss(y_real, y_pred):
    d_ra_real = rel_avg_loss(y_real, y_pred)
    d_ra_fake = rel_avg_loss(y_pred, y_real)

    print(d_ra_real, d_ra_fake)

    y_real = K.concatenate([K.zeros(shape=K.shape(d_ra_real)), K.ones(shape=K.shape(d_ra_fake))], axis=0)
    y_pred = K.concatenate([d_ra_real, d_ra_fake], axis=0)

    return K.mean(K.binary_crossentropy(y_real, y_pred), axis=-1)


class SRGAN():
    def __init__(self, parent_dir):
        # Input shape
        self.channels = 3
        self.lr_height = 64                 # Low resolution height
        self.lr_width = 64                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*4   # High resolution height
        self.hr_width = self.lr_width*4     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 1

        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = 'img_align_celeba'
        self.parent_dir = parent_dir
        self.data_loader = DataLoader(dataset_name=self.dataset_name, parent_dir=self.parent_dir,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss=d_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features, validity, fake_hr])
        self.combined.compile(loss=['binary_crossentropy', 'mse', g_loss, 'mae'],
                              loss_weights=[1e-3, 1, 5e-3, 1e-2],
                              optimizer=optimizer)


    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        vgg.layers[9].activation = None
        # Set outputs to outputs of last conv. layer in block 3
        # See architecture at: https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)

        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            concatenated_inputs = layer_input

            for _ in range(3):
                d = Conv2D(filters, kernel_initializer=SmallInitialization(), kernel_size=3, strides=1, padding='same')(concatenated_inputs)
                d = LeakyReLU()(d)
                concatenated_inputs = Concatenate()([concatenated_inputs, d])

            d = Conv2D(filters, kernel_initializer=SmallInitialization(), kernel_size=3, strides=1, padding='same')(concatenated_inputs)
            return d

        def RRDB(layer_input, filters, beta=0.2):
            d_input = layer_input

            for _ in range(3):
                d = residual_block(d_input, filters)
                d = Lambda(lambda x: x * beta)(d)
                d_input = Add()([d_input, d])

            d_input = Lambda(lambda x: x * beta)(d_input)
            d = Add()([d_input, layer_input])

            return d


        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = RRDB(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = RRDB(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        d11 = Dense(1)(d10)

        return Model(d0, d11)

    def train(self, epochs, batch_size=1, sample_interval=50):
        start_time = datetime.datetime.now()

        def lrate_decay(epoch, lrate):
            if epoch % int(2e+5) == 0:
                return lrate * 0.5
            return lrate

        lrate_callback = LearningRateScheduler(lrate_decay)
        lrate_callback.set_model(self.combined)

        def named_logs(model, logs):
            result = {}
            for l in zip(model.metrics_names, logs):
                result[l[0]] = l[1]
            return result

        tb_callback = TensorBoard(batch_size=batch_size, write_grads=True, write_images=True)
        tb_callback.set_model(self.combined)

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        y_true = np.vstack((valid, fake))

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------
            lrate_callback.on_epoch_begin(epoch)
            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)
            imgs = np.vstack((imgs_hr, fake_hr))

            # Train the discriminators (original images = real / generated = Fake)
            d_loss = self.discriminator.train_on_batch(imgs, y_true)
            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            imgs_hr_pred = self.discriminator.predict(imgs_hr)

            print(imgs_hr_pred.shape)
            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features, imgs_hr_pred, imgs_hr])
            lrate_callback.on_epoch_end(epoch + 1)
            tb_callback.on_epoch_end(epoch, named_logs(self.combined, g_loss))

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("{} time: {}, loss: {}".format(epoch, elapsed_time, g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

        tb_callback.on_train_end()

    def sample_images(self, epoch):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (self.dataset_name, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (self.dataset_name, epoch, i))
            plt.close()

if __name__ == '__main__':
    gan = SRGAN(sys.argv[1] if len(sys.argv) == 2 else '.')
    gan.train(epochs=30000, batch_size=4, sample_interval=50)
