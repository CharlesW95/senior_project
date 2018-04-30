# This represents a new approach, where we feed style activations into the decoder.
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

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Extra image resizing
from adain.image import scale_image

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
        num_epochs=244,
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
    style_encoded = tf.placeholder(tf.float32, shape=encoder_layer_shape) # conv4_1
    output_encoded = adain(content_encoded, style_encoded)

    # TRIVIAL MASK
    trivial_mask_value = gen_trivial_mask()
    trivial_mask = tf.constant(trivial_mask_value, dtype=tf.bool, name="trivial_mask")

    window_mask_value = gen_window_mask()
    window_mask = tf.constant(window_mask_value, dtype=tf.bool, name="window_mask")

    # The same layers we pass in to the decoder need to be the same ones we use
    # to compute loss later.

    # Concatenate relevant inputs to be passed into decoder.
    output_combined = tf.concat([output_encoded, style_encoded], axis=1)
    images = build_decoder(output_combined, weights=None, trainable=True,
        activation=decoder_activation)
    
    with open_weights(vgg) as w:
        vgg = build_vgg(images, w, last_layer=encoder_layer)
        encoder = vgg[encoder_layer]

    # loss setup
    # content_target, style_targets will hold activations of content and style
    # images respectively
    content_layer = vgg[content_layer] # In this case it's the same as encoder_layer
    content_target = tf.placeholder(tf.float32, shape=encoder_layer_shape)
    style_layers = {layer: vgg[layer] for layer in style_layers}

    conv3_1_output_width_t, conv4_1_output_width_t = tf.shape(style_layers["conv3_1"], \
        out_type=tf.int32), tf.shape(style_layers["conv4_1"], out_type=tf.int32)

    style_targets = {
        layer: tf.placeholder(tf.float32, shape=style_layers[layer].shape)
        for layer in style_layers
    }

    conv3_1_output_width = tf.placeholder(tf.int32, shape=(), name="conv3_1_output_width")
    conv4_1_output_width = tf.placeholder(tf.int32, shape=(), name="conv4_1_output_width")

    content_loss = build_content_loss(content_layer, content_target, 0.75)

    style_texture_losses = build_style_texture_losses(style_layers, style_targets, style_weight * 0.1 * 2.0)
    style_content_loss = build_style_content_loss_guided(style_layers, style_targets, output_encoded, trivial_mask, window_mask, 1.0)

    loss = tf.reduce_sum(list(style_texture_losses.values())) + style_content_loss

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

                # step 2
                # run the output batch through the decoder, compute loss
                feed_dict = {
                    output_encoded: output_batch_encoded,
                    style_encoded: style_batch_encoded,
                    # "We use the AdaIN output as the content target, instead of
                    # the commonly used feature responses of the content image"
                    content_target: output_batch_encoded
                    # filtered_x_target: filt_x_targ,
                    # filtered_y_target: filt_y_targ,
                    # conv3_1_output_width: conv3_1_shape[2],
                    # conv4_1_output_width: conv4_1_shape[2]
                }

                for layer in style_targets:
                    feed_dict[style_targets[layer]] = style_target_vals[layer]

                fetches = [train_op, loss, content_loss, style_texture_losses,
                    style_content_loss, tv_loss, global_step]
                result = sess.run(fetches, feed_dict=feed_dict)
                _, loss_val, content_loss_val, style_texture_loss_vals, style_content_loss_val, tv_loss_val, i = result

                # Print out the masks
                # fig = plt.figure()
                # for k in range(8):
                #     mask = fg_val[k, 0, :, :]
                #     pd.DataFrame(mask).to_csv("/output/fg_mask_" + str(k) + ".csv")
                #     fig.add_subplot(2, 4, k+1)
                #     plt.imshow(mask, cmap='gray')
                # plt.savefig("/output/fg_masks_" + str(i) + ".eps", format="eps", dpi=75)

                # fig = plt.figure()
                # for k in range(8):
                #     mask = bg_val[k, 0, :, :]
                #     pd.DataFrame(mask).to_csv("/output/bg_mask_" + str(k) + ".csv")
                #     fig.add_subplot(2, 4, k+1)
                #     plt.imshow(mask, cmap='gray')
                # plt.savefig("/output/bg_masks_" + str(i) + ".eps", format="eps", dpi=75)
                # for k in range(8):
                #     mask = tar_val[k, 0, :, :]
                #     fig.add_subplot(2, 4, k+1)
                #     mask_flattened = mask.flatten()
                #     print("Here is the shape")
                #     print(mask_flattened.shape)
                #     print(mask_flattened[:10])
                #     plt.hist(mask_flattened)
                #     plt.show()
                # plt.savefig("/output/first_layer_hist" + str(i) + ".eps", format="eps", dpi=75)
                # for k in range(8):
                #     mask = tar_val[k, 1, :, :]
                #     fig.add_subplot(2, 4, k+1)
                #     mask_flattened = mask.flatten()
                #     plt.hist(mask_flattened)
                #     plt.show()
                # plt.savefig("/output/second_layer_hist" + str(i) + ".eps", format="eps", dpi=75)
                # for k in range(8):
                #     first_activation = tar_val[k, 0, :, :]
                #     second_activation = tar_val[k, 1, :, :]
                #     pd.DataFrame(first_activation).to_csv("/output/first_activation_" + str(k) + ".csv")
                #     pd.DataFrame(second_activation).to_csv("/output/second_activation_" + str(k) + ".csv")
            
                if i % print_every == 0:
                    style_texture_loss_val = sum(style_texture_loss_vals.values())
                    # style_loss_vals = '\t'.join(sorted(['%s = %0.4f' % (name, val) for name, val in style_loss_vals.items()]))
                    print(i,
                        'loss = %0.4f' % loss_val,
                        'content = %0.4f' % content_loss_val,
                        'style_texture = %0.4f' % style_texture_loss_val,
                        'style_content = %0.4f' % style_content_loss_val,
                        'tv = %0.4f' % tv_loss_val, sep='\t')

                if i % save_every == 0:
                    print('Saving checkpoint')
                    saver.save(sess, os.path.join(checkpoint_dir, 'adain'), global_step=i)

        coord.join(threads)
        saver.save(sess, os.path.join(checkpoint_dir, 'adain-final'))

def visualizeActivations(layerOutput, plotName="figure"):
    fig = plt.figure()
    for i in range(min(4, layerOutput.shape[1])):
        output = layerOutput[0, i, :, :]
        fig.add_subplot(2, 2, i+1)
        plt.imshow(output)
        df =  pd.DataFrame(data=output)
        df.to_csv("/output/%s_%s.csv" % (str(i), plotName))
    plt.savefig("/output/" + plotName + ".eps", format="eps", dpi=75)

# Simple Euclidean distance
def build_content_loss(current, target, weight):
    loss = tf.reduce_mean(tf.squared_difference(current, target))
    loss *= weight
    return loss

def build_style_texture_losses(current_layers, target_layers, weight, epsilon=1e-6):
    losses = {}
    layer_weights = [0.5, 0.75, 1.5, 2.0]
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


def build_style_texture_losses_guided(current_layers, target_layers, weight, mask, epsilon=1e-6):
    losses = {}
    layer_weights = [0.5, 0.75, 1.5, 2.0]
    for i, layer in enumerate(current_layers):
        current, target = current_layers[layer], target_layers[layer]

        # Use the trivial mask to perform some kind of boolean masking

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

# Shape of (8, 1, 32, 32)
def gen_trivial_mask():
    vals_left = [[[[True] * 16] * 32] * 512] * 8
    vals_left = np.array(vals_left)

    vals_right = [[[[True] * 16] * 32] * 512] * 8
    vals_right = np.array(vals_right)

    vals = np.concatenate((vals_left, vals_right), axis=3)
    return vals

# Window mask is useful for masking texture
def gen_window_mask():
    vals = [[[[False] * 32] * 32] * 512] * 8
    vals = np.array(vals)
    vals[:, :, 4:28, 4:28] = True
    return vals

def build_style_content_loss_guided(current_layers, target_layers, content_encoding, trivial_mask, window_mask, weight):
    global output_width_name

    cos_layers = ["conv4_1"]
    output_width_names = ["conv4_1_output_width"]

    style_content_loss = 0.0

    for i, layer in enumerate(cos_layers):
        output_width_name = output_width_names[i] # Set the global variable
        current, target = current_layers[layer], target_layers[layer]

        style_mask = window_mask
        content_mask = tf.logical_not(style_mask)

        # Compute squared differences of activations
        output_style_diff_sq = tf.squared_difference(current, target)
        output_content_diff_sq = tf.squared_difference(current, content_encoding)
        
        output_style_relevant = tf.boolean_mask(output_style_diff_sq, style_mask)
        # output_content_relevant = tf.boolean_mask(output_content_diff_sq, content_mask)
        output_content_relevant = output_content_diff_sq
        
        # Aggregate to obtain loss term
        layer_loss = tf.reduce_mean(output_style_relevant) + tf.reduce_mean(output_content_relevant) * 2.0
        style_content_loss += layer_loss * weight
    
    return style_content_loss

def mapped_bool_generator(filters):
    # Passed in per input, of shape (n_filters * h * w)
    # Return a list of bools (each one corresponding to 1 filter)
    filts = tf.map_fn(is_filter_output_valid, filters, dtype=tf.bool)
    return filts

def is_filter_output_valid(filt):
    # Determine whether we have a negative mapping
    inner_width = tf.constant(5, dtype=tf.int32)
    total_threshold = 2.0
    filtWidth = retrieve_relevant_placholder()
    inner_sum = extract_inner_sum(filt, filtWidth, inner_window_width=inner_width)
    value = inner_sum > total_threshold
    # Create inner matrix to return
    return tf.fill([filtWidth, filtWidth], value)

def extract_inner_sum(tensor, filtWidth, inner_window_width):
    start = (filtWidth - inner_window_width) // 2
    sliced = tf.slice(tensor, begin=(start, start), size=(inner_window_width, inner_window_width))
    return tf.reduce_sum(sliced)

def retrieve_relevant_placholder():
    global output_width_name
    graph = tf.get_default_graph()
    return graph.get_tensor_by_name(output_width_name + ":0")

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
