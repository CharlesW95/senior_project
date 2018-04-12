import tensorflow as tf

def create_sobel_filters():
    sobel_y = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
    sobel_y_filter = tf.reshape(sobel_y, [3, 3, 1, 1])
    sobel_x_filter = tf.transpose(sobel_y_filter, [1, 0, 2, 3])
    return sobel_x_filter, sobel_y_filter

def edge_detection(images):
    x_filter, y_filter = create_sobel_filters()
    filtered_x = tf.nn.conv2d(images, x_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
    filtered_y = tf.nn.conv2d(images, y_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
    return filtered_x, filtered_y
