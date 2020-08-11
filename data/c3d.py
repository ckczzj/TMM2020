import os

import PIL.Image as Image
import cv2
import numpy as np
import tensorflow as tf

np_mean = np.load('crop_mean.npy').reshape([16, 112, 112, 3])


def extract_video_gifs(video_path):
    buf = []
    all_frame = []
    crop_size = 112

    cap = cv2.VideoCapture(video_path)

    res, frame = cap.read()
    while res:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = np.asarray(frame).astype(np.float32)[..., ::-1]
        image = Image.fromarray(image.astype(np.uint8))
        if image.width > image.height:
            scale = float(crop_size) / float(image.height)
            image = np.array(cv2.resize(np.array(image),
                                        (int(image.width * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(image.width)
            image = np.array(cv2.resize(np.array(image),
                                        (crop_size, int(image.height * scale + 1)))).astype(np.float32)
        crop_x = int((image.shape[0] - crop_size) / 2)
        crop_y = int((image.shape[1] - crop_size) / 2)
        image = image[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        # image = cv2.resize(image, (112, 112))
        all_frame.append(image)
        res, frame = cap.read()

    for fc in range(0, len(all_frame) - 16, 8):
        buf.append(all_frame[fc:fc + 16] - np_mean)
    buf = np.array(buf, dtype=np.float32)
    return buf


def conv3d(name, l_input, w, b):
    return tf.nn.bias_add(tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'), b)


def max_pool(name, l_input, k):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)


NUM_CLASSES = 487
CHANNELS = 3
NUM_FRAMES_PER_CLIP = 16


def placeholder_inputs():
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 16, 112, 112, 3))
    labels_placeholder = tf.placeholder(tf.int64, shape=(None))
    return images_placeholder, labels_placeholder


def _variable_with_weight_decay(name, shape, stddev, wd):
    var = tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('losses', weight_decay)
    return var


def inference_c3d(_X, _dropout, _weights, _biases):
    with tf.device("/gpu:0"):
        conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
        conv1 = tf.nn.relu(conv1, 'relu1')
        pool1 = max_pool('pool1', conv1, k=1)

        conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
        conv2 = tf.nn.relu(conv2, 'relu2')
        pool2 = max_pool('pool2', conv2, k=2)

        conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
        conv3 = tf.nn.relu(conv3, 'relu3a')
        conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
        conv3 = tf.nn.relu(conv3, 'relu3b')
        pool3 = max_pool('pool3', conv3, k=2)

        conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
        conv4 = tf.nn.relu(conv4, 'relu4a')
        conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
        conv4 = tf.nn.relu(conv4, 'relu4b')
        pool4 = max_pool('pool4', conv4, k=2)

        conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
        conv5 = tf.nn.relu(conv5, 'relu5a')
        conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
        conv5 = tf.nn.relu(conv5, 'relu5b')
        pool5 = max_pool('pool5', conv5, k=2)

        # pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
        dense1 = tf.reshape(pool5, [-1, _weights['wd1'].get_shape().as_list()[
            0]])  # Reshape conv3 output to fit dense layer input
        dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

        dense1 = tf.nn.relu(dense1, name='fc1')  # Relu activation
        dense11 = tf.nn.dropout(dense1, _dropout)

        dense2 = tf.nn.relu(tf.matmul(dense11, _weights['wd2']) + _biases['bd2'], name='fc2')  # Relu activation
        dense2 = tf.nn.dropout(dense2, _dropout)  # [None, dim]

        out = tf.matmul(dense2, _weights['out']) + _biases['out']
    return dense1


images_placeholder, labels_placeholder = placeholder_inputs()
with tf.variable_scope('var_name') as var_scope:
    weights = {
        'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
        'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
        'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
        'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
        'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
        'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
        'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
        'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
        'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
        'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
        'out': _variable_with_weight_decay('wout', [4096, NUM_CLASSES], 0.04, 0.005)
    }
    biases = {
        'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
        'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
        'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
        'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
        'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
        'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
        'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
        'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
        'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
        'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
        'out': _variable_with_weight_decay('bout', [NUM_CLASSES], 0.04, 0.0),
    }

mat = inference_c3d(images_placeholder, 1.0, weights, biases)

model_name = 'conv3d_deepnetA_sport1m_iter_1900000_TF.model'
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)
init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, model_name)


def extract_feature(sess, matrix):
    frames_num = matrix.shape[0]
    batch_size = 40
    batch_num = int((frames_num - 1) / batch_size + 1)

    result_npy = np.zeros([frames_num, 4096], dtype=np.float)
    for i in range(batch_num):
        temp_size = min(frames_num - i * batch_size, batch_size)
        temp = matrix[i * batch_size:i * batch_size + temp_size, :, :, :]
        # [out_mat] = sess.run([mat], feed_dict={images_placeholder: temp})
        out_mat = mat.eval(session=sess, feed_dict={images_placeholder: temp})
        out_mat = np.array(out_mat)
        result_npy[i * batch_size:i * batch_size + temp_size, :] = out_mat
    return result_npy


# extract features
path = './videos'
for video_name in os.listdir(path):
    video_path = os.path.join(path, video_name)
    save_path = './feature/%s.npy' % video_name[:-4]
    feature = np.load(save_path)
    print(video_path)
    if feature.shape[0] > 0:
        print(feature.shape)
        continue
    buf = extract_video_gifs(video_path)
    print(buf.shape)
    feature = extract_feature(sess, buf)
    print(feature.shape)
    np.save(save_path, feature)
