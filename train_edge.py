#!/usr/bin/env python
from math import ceil, floor
from random import uniform
import argparse
import os

import tensorflow as tf

from adain.nn import build_vgg, vgg_layer_params, build_decoder
from adain.norm import adain
from adain.util import get_params
from adain.weights import open_weights
from edge_detectors import edge_detection

# Extra image resizing
import numpy as np
import matplotlib
matplotlib.use('Agg')
from adain.image import scale_image
import matplotlib.pyplot as plt

def train(
        content_dir='/floyd_images/',
        style_dir='/floyd_images/',
        checkpoint_dir='output',
        decoder_activation='relu',
        initial_size=512,
        random_crop_size=256,
        resume=False,
        optimizer='adam',
        learning_rate=1e-4,
        learning_rate_decay=5e-5,
        momentum=0.9,
        batch_size=8,
        num_epochs=100,
        content_layer='conv4_1',
        style_layers='conv1_1,conv2_1,conv3_1,conv4_1',
        tv_weight=0,
        style_weight=1e-2,
        content_weight=0.75,
        save_every=10000,
        print_every=10,
        gpu=0,
        vgg='/floyd_models/vgg19_weights_normalized.h5'):
    assert initial_size >= random_crop_size, 'Images are too small to be cropped'
    assert gpu >= 0, 'CPU mode is not supported'

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    if not os.path.exists(checkpoint_dir):
        print('Creating checkpoint dir at', checkpoint_dir)
        os.mkdir(checkpoint_dir)

    style_layers = style_layers.split(',')

    # the content layer is also used as the encoder layer
    encoder_layer = content_layer
    encoder_layer_filters = vgg_layer_params(encoder_layer)['filters'] # Just gives you the number of filters
    encoder_layer_shape = (None, encoder_layer_filters, None, None)

    # decoder->encoder setup
    if decoder_activation == 'relu':
        decoder_activation = tf.nn.relu
    elif decoder_activation == 'elu':
        decoder_activation = tf.nn.elu
    else:
        raise ValueError('Unknown activation: ' + decoder_activation)

    # This is a placeholder because we are going to feed it the output
    # from the encoder defined below.
    content_encoded = tf.placeholder(tf.float32, shape=encoder_layer_shape)
    style_encoded = tf.placeholder(tf.float32, shape=encoder_layer_shape)
    output_encoded = adain(content_encoded, style_encoded)
    # NOTE: "images" contains the output of the decoder
    images = build_decoder(output_encoded, weights=None, trainable=True,
        activation=decoder_activation)

    # New placeholder just to hold content images
    # content_image = tf.placeholder(tf.float32, shape=(None, 3, random_crop_size, random_crop_size))
    images_reshaped = tf.transpose(images, perm=(0, 2, 3, 1))
    grayscaled_content = tf.image.rgb_to_grayscale(images_reshaped)
    # Run sobel operators on it
    filtered_x, filtered_y = edge_detection(grayscaled_content)
    
    with open_weights(vgg) as w:
        # We need the VGG for loss computation
        vgg = build_vgg(images, w, last_layer=encoder_layer)
        encoder = vgg[encoder_layer]

    # loss setup
    # content_target, style_targets will hold activations of content and style
    # images respectively
    content_layer = vgg[content_layer] # In this case it's the same as encoder_layer
    content_target = tf.placeholder(tf.float32, shape=encoder_layer_shape)
    style_layers = {layer: vgg[layer] for layer in style_layers}
    style_targets = {
        layer: tf.placeholder(tf.float32, shape=style_layers[layer].shape)
        for layer in style_layers
    }

    # Define placeholders for the targets
    filtered_x_target = tf.placeholder(tf.float32, shape=filtered_x.get_shape())
    filtered_y_target = tf.placeholder(tf.float32, shape=filtered_y.get_shape())

    content_general_loss = build_content_general_loss(content_layer, content_target, 0.5)
    content_edge_loss = build_content_edge_loss(filtered_x, filtered_y, filtered_x_target, filtered_y_target, 3.0)
    style_texture_losses = build_style_texture_losses(style_layers, style_targets, style_weight)
    style_content_loss = build_style_content_loss(style_layers, style_targets, 1.5)

    loss = content_general_loss + content_edge_loss + tf.reduce_sum(list(style_texture_losses.values())) + style_content_loss

    if tv_weight:
        tv_loss = tf.reduce_sum(tf.image.total_variation(images)) * tv_weight
    else:
        tv_loss = tf.constant(0, dtype=tf.float32)
    loss += tv_loss

    # training setup
    batch = setup_input_pipeline(content_dir, style_dir, batch_size,
        num_epochs, initial_size, random_crop_size)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    rate = tf.train.inverse_time_decay(learning_rate, global_step,
        decay_steps=1, decay_rate=learning_rate_decay)

    if optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(rate, beta1=momentum)
    elif optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(rate)
    else:
        raise ValueError('Unknown optimizer: ' + optimizer)

    train_op = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())

        if resume:
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, latest)
        else:
            sess.run(tf.global_variables_initializer())
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        with coord.stop_on_exception():
            while not coord.should_stop():
                content_batch, style_batch = sess.run(batch)
               
                # step 1
                # encode content and style images,
                # compute target style activations,
                # run content and style through AdaIN
                content_batch_encoded = sess.run(encoder, feed_dict={
                    images: content_batch
                })

                style_batch_encoded, style_target_vals = sess.run([encoder, style_layers], feed_dict={
                    images: style_batch
                })

                # This is the AdaIN step
                output_batch_encoded = sess.run(output_encoded, feed_dict={
                    content_encoded: content_batch_encoded,
                    style_encoded: style_batch_encoded
                })
                
                # Actual target values for edge loss
                filt_x_targ, filt_y_targ = sess.run([filtered_x, filtered_y], feed_dict={
                    images: content_batch
                })

                # step 2
                # run the output batch through the decoder, compute loss
                feed_dict = {
                    output_encoded: output_batch_encoded,
                    # "We use the AdaIN output as the content target, instead of
                    # the commonly used feature responses of the content image"
                    content_target: output_batch_encoded,
                    filtered_x_target: filt_x_targ,
                    filtered_y_target: filt_y_targ
                }

                for layer in style_targets:
                    feed_dict[style_targets[layer]] = style_target_vals[layer]

                fetches = [train_op, images, loss, content_general_loss, content_edge_loss, style_texture_losses,
                    style_content_loss, tv_loss, global_step]
                result = sess.run(fetches, feed_dict=feed_dict)
                _, output_images, loss_val, content_general_loss_val, content_edge_loss_val, \
                     style_texture_loss_vals, style_content_loss_val, tv_loss_val, i = result

                # Try to plot these out?
                # (8, 256, 256, 1)
                # save_edge_images(filt_x_orig, batch_size, "x_filters")
                # save_edge_images(filt_y_orig, batch_size, "y_filters")
                # original_content_batch = np.transpose(content_batch, axes=(0, 2, 3, 1))
                # save_edge_images(original_content_batch, batch_size, "original_r")
                # exit()
                if i % print_every == 0:
                    style_texture_loss_val = sum(style_texture_loss_vals.values())
                    # style_loss_vals = '\t'.join(sorted(['%s = %0.4f' % (name, val) for name, val in style_loss_vals.items()]))
                    print(i,
                        'loss = %0.4f' % loss_val,
                        'content_general = %0.4f' % content_general_loss_val,
                        'content_edge = %0.4f' % content_edge_loss_val,
                        'style_texture = %0.4f' % style_texture_loss_val,
                        'style_content = %0.4f' % style_content_loss_val,
                        'tv = %0.4f' % tv_loss_val, sep='\t')

                if i % save_every == 0:
                    print('Saving checkpoint')
                    saver.save(sess, os.path.join(checkpoint_dir, 'adain'), global_step=i)

        coord.join(threads)
        saver.save(sess, os.path.join(checkpoint_dir, 'adain-final'))


# Simple Euclidean distance
def build_content_general_loss(current, target, weight):
    return euclidean_distance(current, target, weight)

def build_content_edge_loss(filtered_x, filtered_y, filtered_x_target, filtered_y_target, weight):
    x_dist = euclidean_distance(filtered_x, filtered_x_target, weight)
    y_dist = euclidean_distance(filtered_y, filtered_y_target, weight)
    return x_dist + y_dist

def euclidean_distance(current, target, weight):
    loss = tf.reduce_mean(tf.squared_difference(current, target))
    loss *= weight
    return loss

def build_style_texture_losses(current_layers, target_layers, weight, epsilon=1e-6):
    losses = {}
    layer_weights = [0.5, 0.75, 1.25, 1.5]
    for i, layer in enumerate(current_layers):
        current, target = current_layers[layer], target_layers[layer]

        current_mean, current_var = tf.nn.moments(current, axes=[2,3], keep_dims=True)
        current_std = tf.sqrt(current_var + epsilon)

        target_mean, target_var = tf.nn.moments(target, axes=[2,3], keep_dims=True)
        target_std = tf.sqrt(target_var + epsilon)

        mean_loss = tf.reduce_sum(tf.squared_difference(current_mean, target_mean))
        std_loss = tf.reduce_sum(tf.squared_difference(current_std, target_std))

        # normalize w.r.t batch size
        n = tf.cast(tf.shape(current)[0], dtype=tf.float32)
        mean_loss /= n
        std_loss /= n

        losses[layer] = (mean_loss + std_loss) * weight * layer_weights[i]

    return losses # Returns a dictionary

def build_style_content_loss(current_layers, target_layers, weight):
    cos_layers = ["conv3_1", "conv4_1"]
    style_content_loss = 0.0
    for layer in cos_layers:
        current, target = current_layers[layer], target_layers[layer]
        # threshold = 0.05 # NOTE: We have to test this
        # mask = current > threshold
        # squared_diffs = tf.squared_difference(current, target)
        # masked_squared_diffs = tf.boolean_mask(squared_diffs, mask)
        # layer_loss = tf.reduce_mean(masked_squared_diffs)
        layer_loss = tf.reduce_mean(tf.squared_difference(current, target))
        style_content_loss += layer_loss * weight
    
    return style_content_loss

def setup_input_pipeline(content_dir, style_dir, batch_size,
        num_epochs, initial_size, random_crop_size):
    content = read_preprocess(content_dir, num_epochs, initial_size, random_crop_size, crop_on=False)
    style = read_preprocess(style_dir, num_epochs, initial_size, random_crop_size)
    return tf.train.shuffle_batch([content, style],
        batch_size=batch_size,
        capacity=1000,
        min_after_dequeue=batch_size*2)

def read_preprocess(path, num_epochs, initial_size, random_crop_size, crop_on=True):
    filenames = tf.train.match_filenames_once(os.path.join(path, '*.tfrecords'))
    filename_queue = tf.train.string_input_producer(filenames,
        num_epochs=num_epochs, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized, features={
        'image': tf.FixedLenFeature([], tf.string),
    })

    image = tf.decode_raw(features['image'], tf.uint8)

    # NOTE: By this point, images have already been resized into squares
    # with a simple center crop in the preprocessing stage.
    image.set_shape((3*initial_size*initial_size))

    # NOTE: Random crop step removed.
    # image = random_crop(image, initial_size, random_crop_size)
    # Introducing center crop to hopefully improve situation

    if crop_on:
        image = tf.reshape(image, (3, initial_size, initial_size))
        image = center_crop(image, random_crop_size)
    else: # If we're not cropping, just resize to get the same image size
        image = tf.reshape(image, (initial_size, initial_size, 3)) # H, W, C
        image = tf.image.resize_images(image, size=(random_crop_size, random_crop_size), align_corners=True)
        image = tf.reshape(image, (3, random_crop_size, random_crop_size)) # Reshape to ensure uniformity

    image = tf.cast(image, tf.float32) / 255
    return image

def random_crop(image, initial_size, crop_size):
    x = ceil(uniform(0, initial_size - crop_size))
    y = ceil(uniform(0, initial_size - crop_size))
    image = image[:,y:y+crop_size,x:x+crop_size]
    image.set_shape((3, crop_size, crop_size))
    return image

# New: replace random_crop with center crop
def center_crop(image, crop_size):
    image_shape = image.get_shape().as_list()
    offset_length = floor(float(crop_size/2))
    x_start = floor(image_shape[2]/2 - offset_length)
    y_start = floor(image_shape[1]/2 - offset_length)
    image = image[:, x_start:x_start+crop_size, y_start:y_start+crop_size]
    image.set_shape((3, crop_size, crop_size))
    return image

def save_edge_images(batch, batch_size, plotname):
    fig = plt.figure()
    for i in range(batch_size):
        image = batch[i, :, :, 0]
        fig.add_subplot(batch_size//2, 2, i+1)
        plt.imshow(image, cmap='gray')
    plt.savefig("/output/" + plotname + ".eps", format="eps", dpi=75)

if __name__ == '__main__':
    params = get_params(train)

    parser = argparse.ArgumentParser(description='AdaIN Style Transfer Training')

    # general
    parser.add_argument('--content_dir', default=params['content_dir'],
        help='A directory with TFRecords files containing content images for training')
    parser.add_argument('--style_dir', default=params['style_dir'],
        help='A directory with TFRecords files containing style images for training')
    parser.add_argument('--vgg', default=params['vgg'],
        help='Path to the weights of the VGG19 network')
    parser.add_argument('--checkpoint_dir', default=params['checkpoint_dir'],
        help='Name of the checkpoint directory')
    parser.add_argument('--decoder_activation', default=params['decoder_activation'],
        help='Activation function in the decoder')
    parser.add_argument('--gpu', default=params['gpu'], type=int,
        help='Zero-indexed ID of the GPU to use')

    # preprocessing
    parser.add_argument('--initial_size', default=params['initial_size'],
        type=int, help='Initial size of training images')
    parser.add_argument('--random_crop_size', default=params['random_crop_size'], type=int,
        help='Images will be randomly cropped to this size')

    # training options
    parser.add_argument('--resume', action='store_true',
        help='If true, resume training from the last checkpoint')
    parser.add_argument('--optimizer', default=params['optimizer'],
        help='Optimizer used, adam or SGD')
    parser.add_argument('--learning_rate', default=params['learning_rate'],
        type=float, help='Learning rate')
    parser.add_argument('--learning_rate_decay', default=params['learning_rate_decay'],
        type=float, help='Learning rate decay')
    parser.add_argument('--momentum', default=params['momentum'],
        type=float, help='Momentum')
    parser.add_argument('--batch_size', default=params['batch_size'],
        type=int, help='Batch size')
    parser.add_argument('--num_epochs', default=params['num_epochs'],
        type=int, help='Number of epochs')
    parser.add_argument('--content_layer', default=params['content_layer'],
        help='Target content layer used to compute the loss')
    parser.add_argument('--style_layers', default=params['style_layers'],
        help='Target style layers used to compute the loss')
    parser.add_argument('--tv_weight', default=params['tv_weight'],
        type=float, help='Weight of the Total Variation loss')
    parser.add_argument('--style_weight', default=params['style_weight'],
        type=float, help='Weight of style loss')
    parser.add_argument('--content_weight', default=params['content_weight'],
        type=float, help='Weight of content loss')

    parser.add_argument('--save_every', default=params['save_every'],
        type=int, help='Save interval')
    parser.add_argument('--print_every', default=params['print_every'],
        type=int, help='Print interval')

    args = parser.parse_args()
    train(**vars(args))
