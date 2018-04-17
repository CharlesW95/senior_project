#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')

from itertools import product
import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pylab 
import scipy.stats as stats

from adain.image import load_image, prepare_image, load_mask, save_image
from adain.coral import coral
from adain.nn import build_vgg, build_decoder
from adain.norm import adain
from adain.weights import open_weights
from adain.util import get_filename, get_params, extract_image_names_recursive

from image_preprocess import center_crop_np

def style_transfer(
        content=None,
        content_dir=None,
        content_size=512,
        style=None,
        style_dir=None,
        style_size=512,
        crop=None,
        preserve_color=None,
        alpha=1.0,
        style_interp_weights=None,
        mask=None,
        output_dir='/output',
        save_ext='jpg',
        gpu=0,
        vgg_weights='/floyd_models/vgg19_weights_normalized.h5',
        decoder_weights='/floyd_models/decoder_weights.h5',
        tf_checkpoint_dir=None):
    assert bool(content) != bool(content_dir), 'Either content or content_dir should be given'
    assert bool(style) != bool(style_dir), 'Either style or style_dir should be given'

    if not os.path.exists(output_dir):
        print('Creating output dir at', output_dir)
        os.mkdir(output_dir)

    # Assume that it is either an h5 file or a name of a TensorFlow checkpoint
    # NOTE: For now, artificially switching off pretrained h5 weights
    decoder_in_h5 = False
    # decoder_weights.endswith('.h5')

    if content:
        content_batch = [content]
    else:
        assert mask is None, 'For spatial control use the --content option'
        content_batch = extract_image_names_recursive(content_dir)

    if style:
        style = style.split(',')
        if mask:
            assert len(style) == 2, 'For spatial control provide two style images'
            style_batch = [style]
        elif len(style) > 1: # Style blending
            if not style_interp_weights:
                # by default, all styles get equal weights
                style_interp_weights = np.array([1.0/len(style)] * len(style))
            else:
                # normalize weights so that their sum equals to one
                style_interp_weights = [float(w) for w in style_interp_weights.split(',')]
                style_interp_weights = np.array(style_interp_weights)
                style_interp_weights /= np.sum(style_interp_weights)
                assert len(style) == len(style_interp_weights), """--style and --style_interp_weights must have the same number of elements"""
            style_batch = [style]
        else:
            style_batch = style
    else:
        assert mask is None, 'For spatial control use the --style option'
        style_batch = extract_image_names_recursive(style_dir)

    print('Number of content images:', len(content_batch))
    print('Number of style images:', len(style_batch))

    if gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
        data_format = 'channels_first'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        data_format = 'channels_last'

    image, content, style, target, encoder, decoder, otherLayers = _build_graph(vgg_weights,
        decoder_weights if decoder_in_h5 else None, alpha, data_format=data_format)

    with tf.Session() as sess:
        if decoder_in_h5:
            sess.run(tf.global_variables_initializer())
        elif tf_checkpoint_dir is not None: # Some checkpoint was provided
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(tf_checkpoint_dir, 'adain-final'))

        for content_path, style_path in product(content_batch, style_batch):
            content_name = get_filename(content_path)
            content_image = load_image(content_path, content_size, crop)

            if isinstance(style_path, list): # Style blending/Spatial control
                style_paths = style_path
                style_name = '_'.join(map(get_filename, style_paths))

                # Gather all style images in one numpy array in order to get
                # their activations in one pass
                style_images = None
                for i, style_path in enumerate(style_paths):
                    style_image = load_image(style_path, style_size, crop)
                    if preserve_color:
                        style_image = coral(style_image, content_image)
                    style_image = prepare_image(style_image)
                    if style_images is None:
                        shape = tuple([len(style_paths)]) + style_image.shape
                        style_images = np.empty(shape)
                    assert style_images.shape[1:] == style_image.shape, """Style images must have the same shape"""
                    style_images[i] = style_image
                style_features = sess.run(encoder, feed_dict={
                    image: style_images
                })

                content_image = prepare_image(content_image)
                content_feature = sess.run(encoder, feed_dict={
                    image: content_image[np.newaxis,:]
                })

                if mask:
                    # For spatial control, extract foreground and background
                    # parts of the content using the corresponding masks,
                    # run them individually through AdaIN then combine
                    if data_format == 'channels_first':
                        _, c, h, w = content_feature.shape
                        content_view_shape = (c, -1)
                        mask_shape = lambda mask: (c, len(mask), 1)
                        mask_slice = lambda mask: (slice(None),mask)
                    else:
                        _, h, w, c = content_feature.shape
                        content_view_shape = (-1, c)
                        mask_shape = lambda mask: (1, len(mask), c)
                        mask_slice = lambda mask: (mask,slice(None))

                    mask = load_mask(mask, h, w).reshape(-1)
                    fg_mask = np.flatnonzero(mask == 1)
                    bg_mask = np.flatnonzero(mask == 0)

                    content_feat_view = content_feature.reshape(content_view_shape)
                    content_feat_fg = content_feat_view[mask_slice(fg_mask)].reshape(mask_shape(fg_mask))
                    content_feat_bg = content_feat_view[mask_slice(bg_mask)].reshape(mask_shape(bg_mask))

                    style_feature_fg = style_features[0]
                    style_feature_bg = style_features[1]

                    target_feature_fg = sess.run(target, feed_dict={
                        content: content_feat_fg[np.newaxis,:],
                        style: style_feature_fg[np.newaxis,:]
                    })
                    target_feature_fg = np.squeeze(target_feature_fg)

                    target_feature_bg = sess.run(target, feed_dict={
                        content: content_feat_bg[np.newaxis,:],
                        style: style_feature_bg[np.newaxis,:]
                    })
                    target_feature_bg = np.squeeze(target_feature_bg)

                    target_feature = np.zeros_like(content_feat_view)
                    target_feature[mask_slice(fg_mask)] = target_feature_fg
                    target_feature[mask_slice(bg_mask)] = target_feature_bg
                    target_feature = target_feature.reshape(content_feature.shape)
                else:
                    # For style blending, get activations for each style then
                    # take a weighted sum.
                    target_feature = np.zeros(content_feature.shape)
                    for style_feature, weight in zip(style_features, style_interp_weights):
                        target_feature += sess.run(target, feed_dict={
                            content: content_feature,
                            style: style_feature[np.newaxis,:]
                        }) * weight
            else:
                # NOTE: This is the part we care about, if only 1 style image is provided.
                style_name = get_filename(style_path)
                style_image = load_image(style_path, style_size, crop=True) # This only gives us square crop
                # style_image = center_crop_np(style_image) # Actually crop the center out

                if preserve_color:
                    style_image = coral(style_image, content_image)
                style_image = prepare_image(style_image, True, data_format)
                content_image = prepare_image(content_image, True, data_format)
                
                # Extract other layers
                conv3_1_layer, conv4_1_layer = otherLayers
                style_feature, conv3_1_out_style, conv4_1_out_style = sess.run([encoder, conv3_1_layer, conv4_1_layer], feed_dict={
                    image: style_image[np.newaxis,:]
                })

                content_feature = sess.run(encoder, feed_dict={
                    image: content_image[np.newaxis,:]
                })

                target_feature = sess.run(target, feed_dict={
                    content: content_feature,
                    style: style_feature
                })

            output = sess.run(decoder, feed_dict={
                content: content_feature,
                target: target_feature
            })

            # Grab the relevant layer outputs to see what's being minimized.
            conv3_1_out_output, conv4_1_out_output = sess.run([conv3_1_layer, conv4_1_layer], feed_dict={
                image: output
            })

            filename = '%s_stylized_%s.%s' % (content_name, style_name, save_ext)
            filename = os.path.join(output_dir, filename)
            save_image(filename, output[0], data_format=data_format)
            print('Output image saved at', filename)

            # TODO: Change these layers.
            layersToViz = [conv3_1_out_style, conv4_1_out_style, conv3_1_out_output, conv4_1_out_output]

            for i, layer in enumerate(layersToViz):
                visualizeActivations(layer, plotName=str(i+1))


def visualizeActivations(layerOutput, plotName="figure"):
    fig = plt.figure()
    for i in range(min(64, layerOutput.shape[1])):
        output = layerOutput[0, i, :, :]
        fig.add_subplot(8, 8, i+1)
        # plt.imshow(output)
        plt.hist(output)
        plt.show()
    plt.savefig("/output/" + plotName + ".eps", format="eps", dpi=75)

def testNormality(layerOutput):
    p_values = []
    # Collect p-value for every feature map.
    for i in range(layerOutput.shape[1]):
        flattened_output = layerOutput[0, i, :, :].flatten()
        k2, p = stats.normaltest(flattened_output)
        # Collect statistics
        p_values.append(p)

    # Run aggregation on entire array
    p_values = np.asarray(p_values)
    print(np.mean(p_values))

def _build_graph(vgg_weights, decoder_weights, alpha, data_format):
    if data_format == 'channels_first':
        image = tf.placeholder(shape=(None,3,None,None), dtype=tf.float32)
        content = tf.placeholder(shape=(1,512,None,None), dtype=tf.float32)
        style = tf.placeholder(shape=(1,512,None,None), dtype=tf.float32)
    else:
        image = tf.placeholder(shape=(None,None,None,3), dtype=tf.float32)
        content = tf.placeholder(shape=(1,None,None,512), dtype=tf.float32)
        style = tf.placeholder(shape=(1,None,None,512), dtype=tf.float32)

    target = adain(content, style, data_format=data_format)
    weighted_target = target * alpha + (1 - alpha) * content

    with open_weights(vgg_weights) as w:
        vgg = build_vgg(image, w, data_format=data_format)
        encoder = vgg['conv4_1']

        # Here we can return other layers to check out style encoding
        otherLayerNames = ["conv3_1", "conv4_1"]
        otherLayers = [vgg[i] for i in otherLayerNames]

    if decoder_weights:
        with open_weights(decoder_weights) as w:
            decoder = build_decoder(weighted_target, w, trainable=False,
                data_format=data_format)
    else:
        decoder = build_decoder(weighted_target, None, trainable=False,
            data_format=data_format)

    # Return other layers on top of original outputs
    return image, content, style, target, encoder, decoder, otherLayers


if __name__ == '__main__':
    params = get_params(style_transfer)
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer')

    parser.add_argument('--content', help='File path to the content image')
    parser.add_argument('--content_dir', help="""Directory path to a batch of
        content images""")
    parser.add_argument('--style', help="""File path to the style image,
        or multiple style images separated by commas if you want to do style
        interpolation or spatial control""")
    parser.add_argument('--style_dir', help="""Directory path to a batch of
        style images""")
    parser.add_argument('--vgg_weights', default=params['vgg_weights'],
        help='Path to the weights of the VGG19 network')
    parser.add_argument('--decoder_weights', default=params['decoder_weights'],
        help='Path to the decoder')

    parser.add_argument('--content_size', default=params['content_size'],
        type=int, help="""Maximum size for the content image, keeping
        the original size if set to 0""")
    parser.add_argument('--style_size', default=params['style_size'], type=int,
        help="""Maximum size for the style image, keeping the original
        size if set to 0""")
    parser.add_argument('--crop', action='store_true', help="""If set, center
        crop both content and style image before processing""")
    parser.add_argument('--save_ext', default=params['save_ext'],
        help='The extension name of the output image')
    parser.add_argument('--gpu', default=params['gpu'], type=int,
        help='Zero-indexed ID of the GPU to use; for CPU mode set to -1')
    parser.add_argument('--output_dir', default=params['output_dir'],
        help='Directory to save the output image(s)')

    parser.add_argument('--preserve_color', action='store_true',
        help='If set, preserve color of the content image')
    parser.add_argument('--alpha', default=params['alpha'], type=float,
        help="""The weight that controls the degree of stylization. Should be
        between 0 and 1""")
    parser.add_argument('--style_interp_weights', help="""The weight for
        blending the style of multiple style images""")
    parser.add_argument('--mask', help="""Mask to apply spatial
        control, assume to be the path to a binary mask of the same size as
        content image""")
    parser.add_argument('--tf_checkpoint_dir', default=params['tf_checkpoint_dir'], type=str,
        help="""Location of tf_checkpoint, if provided.""")
    

    args = parser.parse_args()
    style_transfer(**vars(args))
