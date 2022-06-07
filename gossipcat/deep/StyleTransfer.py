#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author:     Ewen Wang
email:      wolfgangwong2012@gmail.com
license:    Apache License 2.0
"""

import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (10,10)
mpl.rcParams['axes.grid'] = False



class StyleTransfer(object):
    """docstring for StyleTransfer"""
    def __init__(self, path_content=None, path_style=None, iterations=1000, 
                  display=True, display_interval=1,
                  weight_content=1e3, weight_style=1e-2):
        super(StyleTransfer, self).__init__()
        self.path_content=path_content
        self.path_style=path_style
        self.iterations=iterations
        self.display=display
        self.display_interval=display_interval
        self.weight_content=weight_content
        self.weight_style=weight_style
        self.loss_weights = (self.weight_style, self.weight_content)

        # load images
        self.content = self.load_resize_img(self.path_content) 
        self.style = self.load_resize_img(self.path_style)

        # initialize
        self.input_vgg_content = self.process_img_as_vgg_input(self.content)
        self.input_vgg_style = self.process_img_as_vgg_input(self.style)
        self.input_vgg_init = self.process_img_as_vgg_input(self.content)
        self.input_vgg_init = tf.Variable(self.input_vgg_init, dtype=tf.float32)

        # model
        self.model = self.setup_model()

        # represent
        self.features_style, self.features_content = self.represent_features()
        self.features_style_gram = [self.gram_matrix(feature_style) for feature_style in self.features_style]

        # optimization initialize
        self.cfg = {
          'model': self.model,
          'loss_weights': self.loss_weights,
          'input_vgg_init': self.input_vgg_init,
          'features_style_gram': self.features_style_gram,
          'features_content': self.features_content
        }
        
        self.best_loss, self.best_img = float('inf'), None
        self.imgs = []

    def load_resize_img(self, path, max_dim=512):
        img = Image.open(path)
        
        long = max(img.size)
        scale = max_dim/long
        
        img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        # We need to broadcast the image array such that it has a batch dimension 
        img_array = np.expand_dims(img, axis=0)
        return img_array

    def display_img(self, img_array, title=None):
        # Remove the batch dimension
        out = np.squeeze(img_array, axis=0)
        # Normalize for display 
        out = out.astype('uint8')
        plt.imshow(out)
        if title:
            plt.title(title)
        return None

    ### input

    def process_img_as_vgg_input(self, img_array):
        img_vgg = tf.keras.applications.vgg19.preprocess_input(img_array)
        return img_vgg

    def deprocess_img(self, img_processed):
        x = img_processed.copy()
        if len(x.shape) == 4:
            x = np.squeeze(x, 0)
        assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")
        if len(x.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")

        # perform the inverse of the preprocessing step
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]

        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def setup_model(self):
        # Content layer where will pull our feature maps
        layers_content = ['block5_conv2'] 

        # Style layer we are interested in
        layers_style = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1']

        self.num_layers_content = len(layers_content)
        self.num_layers_style = len(layers_style)

        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        # Get output layers corresponding to style and content layers 
        outputs_style = [vgg.get_layer(name).output for name in layers_style]
        outputs_content = [vgg.get_layer(name).output for name in layers_content]
        outputs_model = outputs_style + outputs_content

        self.model = tf.keras.Model(inputs=vgg.input, outputs=outputs_model, name='style transfer')
        for layer in self.model.layers:
            layer.trainable = False

        return self.model


    def represent_features(self):
        # batch compute content and style features
        outputs_style = self.model(self.input_vgg_style)
        outputs_content = self.model(self.input_vgg_content)

        # Get the style and content feature representations from our model  
        self.features_style = [style_layer[0] for style_layer in outputs_style[:self.num_layers_style]]
        self.features_content = [content_layer[0] for content_layer in outputs_content[self.num_layers_style:]]
        return self.features_style, self.features_content

    ### loss function

    def gram_matrix(self, input_tensor):
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def get_loss_content(self, base_content, target):
        return tf.reduce_mean(tf.square(base_content - target))

    def get_loss_style(self, base_style, gram_target):
        height, width, channels = base_style.get_shape().as_list()
        gram_style = self.gram_matrix(base_style)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def compute_loss(self):
        # Feed our init image through our model. This will give us the content and 
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        outputs_model = self.model(self.input_vgg_init)

        style_output_features = outputs_model[:self.num_layers_style]
        content_output_features = outputs_model[self.num_layers_style:]

        self.score_style = 0
        self.score_content = 0

        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        weight_per_style_layer = 1.0 / float(self.num_layers_style)
        for target_style, comb_style in zip(self.features_style_gram, style_output_features):
            self.score_style += weight_per_style_layer * self.get_loss_style(comb_style[0], target_style)

        # Accumulate content losses from all layers 
        weight_per_content_layer = 1.0 / float(self.num_layers_content)
        for target_content, comb_content in zip(self.features_content, content_output_features):
            self.score_content += weight_per_content_layer * self.get_loss_content(comb_content[0], target_content)

        self.score_style *= self.weight_style
        self.score_content *= self.weight_content

        # Get total loss
        self.loss_total = self.score_style + self.score_content 
        return self.loss_total, self.score_style, self.score_content

    ## optimization

    def compute_grads(self):
        with tf.GradientTape() as tape: 
            self.losses = self.compute_loss()
            
        # Compute gradients wrt input image
        self.loss_total = self.losses[0]
        return tape.gradient(self.loss_total, self.cfg['input_vgg_init']), self.losses


    def optimize(self, 
                 iterations=1000, learning_rate=5, beta_1=0.99, epsilon=1e-1,
                 display=True, display_interval=1, clear_cache=True):
        
        if clear_cache:
            self.imgs = []
            
        self.iterations = iterations
        self.display = display 
        self.display_interval = display_interval

        # Create our optimizer
        opt = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, epsilon=epsilon)
        
        start_time = time.time()
        global_start = time.time()

        norm_means = np.array([103.939, 116.779, 123.68])
        min_vals = -norm_means
        max_vals = 255 - norm_means 
        
        for i in range(self.iterations):
            self.grads, self.losses = self.compute_grads()
            self.loss, self.score_style, self.score_content = self.losses
            opt.apply_gradients([(self.grads, self.input_vgg_init)])
            clipped = tf.clip_by_value(self.input_vgg_init, min_vals, max_vals)
            self.input_vgg_init.assign(clipped)
            end_time = time.time() 

            if self.loss < self.best_loss:
                # Update best loss and best image from total loss. 
                self.best_loss = self.loss
                self.best_img = self.deprocess_img(self.input_vgg_init.numpy())
            
            if self.display:
                import IPython.display
                if i % self.display_interval== 0:
                    start_time = time.time()

                    # Use the .numpy() method to get the concrete numpy array
                    plot_img = self.input_vgg_init.numpy()
                    plot_img = self.deprocess_img(plot_img)
                    self.imgs.append(plot_img)
                    IPython.display.clear_output(wait=True)
                    IPython.display.display_png(Image.fromarray(plot_img))
                    print('Iteration: {}'.format(i))        
                    print('Total loss: {:.2e}, ' 
                          'style loss: {:.2e}, '
                          'content loss: {:.2e}, '
                          'time: {:.4f}s'.format(self.loss, self.score_style, self.score_content, 
                                                 time.time()-start_time))
        Image.fromarray(self.best_img)
        return None


    def show_results(self, show_large_final=True):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        self.display_img(self.content, 'Content Image')

        plt.subplot(1, 2, 2)
        self.display_img(self.style, 'Style Image')

        if show_large_final: 
            plt.figure(figsize=(10, 10))

            plt.imshow(self.best_img)
            plt.title('Output Image')
            plt.show()
        return None

    def show_inputs(self):
        plt.figure(figsize=(15,15))

        plt.subplot(1, 2, 1)
        self.display_img(self.content, 'Content Image')

        plt.subplot(1, 2, 2)
        self.display_img(self.style, 'Style Image')
        plt.show()
        return None
            