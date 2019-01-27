"""
Super-resolution of CelebA using Generative Adversarial Networks.

The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0

Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to 'datasets/'
4. Run the sript using command 'python srgan.py'
"""

from __future__ import print_function, division
import scipy

from keras.datasets import mnist
#from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add, Layer
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
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

class SRGAN():
    def __init__(self, parent_dir, residual_blocks):
        # Input shape
        self.parent_dir = parent_dir
        self.channels = 3
        self.lr_height = 64                 # Low resolution height
        self.lr_width = 64                  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height*4   # High resolution height
        self.hr_width = self.lr_width*4     # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = residual_blocks

        optimizer = Adam(0.0002, 0.5)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        self.vgg.summary()

        # Configure data loader
        self.dataset_name = 'img_align_celeba'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      parent_dir=self.parent_dir,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['mse', discriminator_loss],
            loss_weights=[1e-3, 1],
            optimizer=optimizer,
            metrics=['accuracy'])

        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()

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
        validity = self.discriminator(fake_hr)[0]

        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=[generator_loss, 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

        self.generator.summary()
        self.combined.summary()


    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
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
            d = ConvSN2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = SelfAttention()(d)
            d = Activation('relu')(d)
          #  d = BatchNormalization(momentum=0.8)(d)
            d = ConvSN2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = SelfAttention()(d)
         #   d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = ConvSN2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = ConvSN2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = SelfAttention()(c1)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = ConvSN2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = SelfAttention()(c2)
     #   c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = ConvSN2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True, sa=True):
            """Discriminator layer"""
            d = ConvSN2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            if sa:
                d = SelfAttention()(d)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False, sa=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df*2)
        d4 = d_block(d3, self.df*2, strides=2)
        d5 = d_block(d4, self.df*4)
        d6 = d_block(d5, self.df*4, strides=2)
        d7 = d_block(d6, self.df*8)
        d8 = d_block(d7, self.df*8, strides=2)

        d9 = DenseSN(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, [validity, validity])

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        labels = np.concatenate((valid, fake))

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            imgs = np.concatenate((imgs_hr, fake_hr))

            # Train the discriminators (original images = real / generated = Fake)
            d_loss = self.discriminator.train_on_batch(imgs, [labels, labels])
            #d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            #d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            #valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

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
    parent_dir = '.'
    residual_blocks = 1
    batch_size = 1

    if len(sys.argv) > 1:
        parent_dir = sys.argv[1]

        if len(sys.argv) > 2:
            residual_blocks = int(sys.argv[2])

            if len(sys.argv) > 3:
                batch_size = int(sys.argv[3])

    print(residual_blocks)
    gan = SRGAN(sys.argv[1], residual_blocks = residual_blocks)
    gan.train(epochs=30000, batch_size=batch_size, sample_interval=50)
