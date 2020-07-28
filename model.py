import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
import util
import layers
import argparse


def load_images(file_name_list, base_dir, use_augmentation=False):
    images = []

    for file_name in file_name_list:
        fullname = os.path.join(base_dir, file_name).replace("\\", "/")
        img = cv2.imread(fullname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for_face = False

        if for_face is False:
            # For SKC
            if use_augmentation is True:
                resized_img = cv2.resize(img, dsize=(512, 520), interpolation=cv2.INTER_CUBIC)
                resized_img_hd = cv2.resize(img, dsize=(1024, 1040), interpolation=cv2.INTER_CUBIC)
        else:
            # For face
            img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

        if img is not None:
            img = np.array(img)

            if for_face is True:
                # For face
                n_img = (img - 128.0) / 128.0
                images.append(n_img)

                if use_augmentation is True:
                    n_img = cv2.flip(img, 1)
                    n_img = (img - 128.0) / 128.0
                    images.append(n_img)
            else:
                img = img[4:260, 0:256]

                # Center crop
                center_x = 256 // 2
                center_y = 256 // 2
                img = img[center_y-64:center_y+64, center_x-64:center_x+64]
                n_img = (img - 128.0) / 128.0
                images.append(n_img)

                if use_augmentation is True:
                    n_img = cv2.flip(img, 1)
                    n_img = (n_img - 128.0) / 128.0
                    images.append(n_img)

                    img = np.array(resized_img)
                    img = img[8:520, 0:512]
                    center_x = 512 // 2
                    center_y = 512 // 2
                    img = img[center_y-64:center_y+64, center_x-64:center_x+64]
                    n_img = (img - 128.0) / 128.0
                    images.append(n_img)

                    img = np.array(resized_img_hd)
                    img = img[16:1040, 0:1024]
                    center_x = 1024 // 2
                    center_y = 1024 // 2
                    img = img[center_y - 64:center_y + 64, center_x - 64:center_x + 64]
                    n_img = (img - 128.0) / 128.0
                    images.append(n_img)

    return np.array(images)


def generator(latent, activation='swish', scope='generator_network', norm='layer', b_train=False, use_upsample=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        block_depth = dense_block_depth * 16

        l = latent

        print('Generator Input: ' + str(latent.get_shape().as_list()))
        transform_channel = latent.get_shape().as_list()[-1] // 16

        l = tf.reshape(l, shape=[-1, 4, 4, transform_channel])

        block_depth = block_depth * 2
        l = layers.conv(l, scope='trans_conv', filter_dims=[1, 1, block_depth], stride_dims=[1, 1], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='trans_norm_1')
        l = act_func(l)
        l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                      act_func=act_func, norm=norm, b_train=b_train, scope='trans_0')

        num_iter = input_width // 4
        num_iter = int(np.sqrt(num_iter))

        for i in range(num_iter):
            block_depth = block_depth // 2

            if use_upsample is True:
                w = l.get_shape().as_list()[2]
                h = l.get_shape().as_list()[1]
                #l = tf.image.resize_bilinear(l, (2 * h, 2 * w))
                l = tf.image.resize_bicubic(l, (2 * h, 2 * w))
                #l = tf.image.resize_nearest_neighbor(l, (2 * h, 2 * w))
                print('Upsampling ' + str(i) + ': ' + str(l.get_shape().as_list()))
                l = layers.conv(l, scope='up_' + str(i), filter_dims=[3, 3, block_depth], stride_dims=[1, 1], non_linear_fn=None)
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='up_norm_' + str(i))
                l = act_func(l)

                for j in range(3):
                    l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                  act_func=act_func, norm=norm, b_train=b_train, use_dilation=False,
                                                  scope='block_' + str(i) + '_' + str(j))
            else:
                l = layers.deconv(l, b_size=l.get_shape().as_list()[0], scope='deconv_' + str(i), filter_dims=[3, 3, block_depth],
                                  stride_dims=[2, 2], padding='SAME', non_linear_fn=None)
                print('Deconvolution ' + str(i) + ': ' + str(l.get_shape().as_list()))
                l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='deconv_norm_' + str(i))
                l = act_func(l)

        if use_upsample is False:
            for i in range(4):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                              act_func=act_func, norm=norm, b_train=b_train, use_dilation=False,
                                              scope='tr_block_' + str(i))

        l = layers.conv(l, scope='last', filter_dims=[1, 1, num_channel], stride_dims=[1, 1], non_linear_fn=tf.nn.tanh, bias=False)

        print('Generator Final: ' + str(l.get_shape().as_list()))

    return l


def discriminator(x, activation='relu', scope='discriminator_network', norm='layer', b_train=False, use_patch=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Encoder Input: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth

        #x = tf.slice(x, [0, 32, 32, 0], [-1, 64, 64, -1])

        l = layers.conv(x, scope='conv0', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm0')
        l = act_func(l)

        if use_patch is True:
            print('Discriminator Block 0: ' + str(l.get_shape().as_list()))

            l = layers.add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                act_func=act_func, norm=norm, b_train=b_train, scope='dense_block_1')

            block_depth = block_depth * 2

            l = layers.conv(l, scope='tr1', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm1')
            l = act_func(l)

            print('Discriminator Block 1: ' + str(l.get_shape().as_list()))

            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='res_block_1_' + str(i))

            # l = layers.self_attention(l, block_depth)

            block_depth = block_depth * 2

            l = layers.conv(l, scope='tr2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm2')
            l = act_func(l)

            print('Discriminator Block 2: ' + str(l.get_shape().as_list()))

            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False,
                                          scope='res_block_2_' + str(i))
            block_depth = block_depth * 2

            l = layers.conv(l, scope='tr3', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm3')
            l = act_func(l)

            print('Discriminator Block 3: ' + str(l.get_shape().as_list()))
            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False,
                                             scope='gp')
            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.global_avg_pool(last_layer, output_length=1, use_bias=False,
                                             scope='gp_logit')
            print('Discriminator Logit Dims: ' + str(logit.get_shape().as_list()))
        else:

            print('Discriminator Block 0: ' + str(l.get_shape().as_list()))

            l = layers.add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                                act_func=act_func, norm=norm, b_train=b_train, scope='dense_block_1')

            block_depth = block_depth * 2

            l = layers.conv(l, scope='tr1', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm1')
            l = act_func(l)

            print('Discriminator Block 1: ' + str(l.get_shape().as_list()))

            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, scope='res_block_1_' + str(i))

            # l = layers.self_attention(l, block_depth)

            block_depth = block_depth * 2

            l = layers.conv(l, scope='tr2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm2')
            l = act_func(l)

            print('Discriminator Block 2: ' + str(l.get_shape().as_list()))

            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, use_dilation=False, scope='res_block_2_' + str(i))
            anchor = l
            block_depth = block_depth * 2

            # [8 x 8]
            l = layers.conv(l, scope='tr3', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm3')
            l = act_func(l)

            print('Discriminator Block 3: ' + str(l.get_shape().as_list()))

            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, use_dilation=False, scope='res_block_3_' + str(i))

            # [4 x 4]
            block_depth = block_depth * 2
            l = layers.conv(l, scope='tr4', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
            l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm4')
            l = act_func(l)

            print('Discriminator Block 4: ' + str(l.get_shape().as_list()))
            l = layers.self_attention(l, block_depth, act_func=act_func)
            for i in range(2):
                l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                              norm=norm, b_train=b_train, use_dilation=False, scope='res_block_4_' + str(i))

            last_layer = l
            feature = layers.global_avg_pool(last_layer, output_length=representation_dim // 8, use_bias=False, scope='gp')

            print('Discriminator GP Dims: ' + str(feature.get_shape().as_list()))

            logit = layers.fc(feature, 1, non_linear_fn=None, scope='flat')

    return feature, logit


def encoder(x, activation='relu', scope='encoder', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Encoder Input: ' + str(x.get_shape().as_list()))
        block_depth = dense_block_depth

        # [128 x 128]
        l = layers.conv(x, scope='conv0', filter_dims=[3, 3, block_depth], stride_dims=[1, 1],
                        non_linear_fn=None, bias=False)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm0')
        l = act_func(l)

        print('Encoder Block 0: ' + str(l.get_shape().as_list()))

        l = layers.add_residual_dense_block(l, filter_dims=[3, 3, block_depth], num_layers=2,
                                            act_func=act_func, norm=norm, b_train=b_train, scope='dense_block_1')

        block_depth = block_depth * 2

        # [64 x 64]
        l = layers.conv(l, scope='tr1', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm1')
        l = act_func(l)

        print('Encoder Block 1: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l,  filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, scope='res_block_1_' + str(i))

        block_depth = block_depth * 2

        # [32 x 32]
        l = layers.conv(l, scope='tr2', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm2')
        l = act_func(l)

        print('Encoder Block 2: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False, scope='res_block_2_' + str(i))
        block_depth = block_depth * 2

        # [16 x 16]
        l = layers.conv(l, scope='tr3', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm3')
        l = act_func(l)

        print('Encoder Block 3: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False, scope='res_block_3_' + str(i))

        # [8 x 8]
        block_depth = block_depth * 2
        l = layers.conv(l, scope='tr4', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm4')
        l = act_func(l)

        print('Encoder Block 4: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False, scope='res_block_4_' + str(i))

        # [4 x 4]
        block_depth = block_depth * 2
        l = layers.conv(l, scope='tr5', filter_dims=[3, 3, block_depth], stride_dims=[2, 2], non_linear_fn=None)
        l = layers.conv_normalize(l, norm=norm, b_train=b_train, scope='norm5')
        l = act_func(l)

        print('Encoder Block 5: ' + str(l.get_shape().as_list()))

        for i in range(2):
            l = layers.add_residual_block(l, filter_dims=[3, 3, block_depth], num_layers=2, act_func=act_func,
                                          norm=norm, b_train=b_train, use_dilation=False,
                                          scope='res_block_5_' + str(i))

        last_layer = l

        context = layers.global_avg_pool(last_layer, output_length=representation_dim, use_bias=False, scope='gp')

        print('Encoder GP Dims: ' + str(context.get_shape().as_list()))

    return context


def latent_discriminator(x, activation='relu', scope='latent_discriminator_network', norm='layer', b_train=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if activation == 'swish':
            act_func = util.swish
        elif activation == 'relu':
            act_func = tf.nn.relu
        elif activation == 'lrelu':
            act_func = tf.nn.leaky_relu
        else:
            act_func = tf.nn.sigmoid

        print('Latent Discriminator Input: ' + str(x.get_shape().as_list()))

        l = x
        l = layers.fc(l, l.get_shape().as_list()[-1] // 2, non_linear_fn=act_func, scope='flat1')
        print('Latent Discriminator layer 1: ' + str(l.get_shape().as_list()))
        #l = layers.layer_norm(l, scope='ln0')

        feature = layers.fc(l, l.get_shape().as_list()[-1] // 4, non_linear_fn=act_func, scope='flat2')
        print('Latent Discriminator Feature: ' + str(feature.get_shape().as_list()))
        logit = layers.fc(feature, 1, non_linear_fn=None, scope='final')

    return feature, logit


def get_feature_matching_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        #loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(target, value)))))
    return gamma * loss


def get_discriminator_loss(real, fake, type='wgan', gamma=1.0):
    if type == 'wgan':
        # wgan loss
        d_loss_real = tf.reduce_mean(real)
        d_loss_fake = tf.reduce_mean(fake)

        # W Distant: f(real) - f(fake). Maximizing W Distant.
        return gamma * (d_loss_fake - d_loss_real), d_loss_real, d_loss_fake
    elif type == 'ce':
        # cross entropy
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake
    elif type == 'hinge':
        d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real))
        d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake))
        return gamma * (d_loss_fake + d_loss_real), d_loss_real, d_loss_fake


def get_residual_loss(value, target, type='l1', gamma=1.0):
    if type == 'rmse':
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(target, value))))
    elif type == 'cross-entropy':
        eps = 1e-10
        loss = tf.reduce_mean(-1 * target * tf.log(value + eps) - 1 * (1 - target) * tf.log(1 - value + eps))
    elif type == 'l1':
        #loss = tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.abs(tf.subtract(target, value)))
    elif type == 'l2':
        #loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(target, value)), [1]))
        loss = tf.reduce_mean(tf.square(tf.subtract(target, value)))

    loss = gamma * loss

    return loss


def get_diff_loss(anchor, positive, negative):
    a_p = get_residual_loss(anchor, positive, 'l1')
    a_n = get_residual_loss(anchor, negative, 'l1')
    # a_n > a_p + margin
    # a_p - a_n + margin < 0
    # minimize (a_p - a_n + margin)
    return tf.reduce_mean(a_p / a_n)


def get_gradient_loss(img1, img2):
    image_a = img1 #tf.expand_dims(img1, axis=0)
    image_b = img2 #tf.expand_dims(img2, axis=0)

    dx_a, dy_a = tf.image.image_gradients(image_a)
    dx_b, dy_b = tf.image.image_gradients(image_b)

    v_a = tf.reduce_mean(tf.image.total_variation(image_a))
    v_b = tf.reduce_mean(tf.image.total_variation(image_b))

    #loss = tf.abs(tf.subtract(v_a, v_b))
    loss = tf.reduce_mean(tf.abs(tf.subtract(dx_a, dx_b))) + tf.reduce_mean(tf.abs(tf.subtract(dy_a, dy_b)))

    return loss


def generate_sample_z(low, high, num_samples, sample_length, b_uniform=True):
    if b_uniform is True:
        z = np.random.uniform(low=low, high=high, size=[num_samples, sample_length])
    else:
        z = np.random.normal(low, high, size=[num_samples, sample_length])

    return z


def make_multi_modal_noise(num_mode=8):
    size = representation_dim // num_mode

    for i in range(batch_size):
        noise = tf.random_normal(shape=[batch_size, size], mean=0.0, stddev=1.0, dtype=tf.float32)

    for i in range(num_mode-1):
        n = tf.random_normal(shape=[batch_size, size], mean=0.0, stddev=1.0, dtype=tf.float32)
        noise = tf.concat([noise, n], axis=1)

    return noise


def train_encoder(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent = encoder(X, activation='relu', norm='instance', b_train=b_train, scope='encoder')
    fake_X = generator(latent, activation='relu', norm='instance', b_train=b_train, scope='generator',
                       use_upsample=True)
    # Adversarial Discriminator
    feature_fake, logit_fake = discriminator(fake_X, activation='swish', norm='instance', b_train=b_train,
                                             scope='discriminator', use_patch=False)
    feature_real, logit_real = discriminator(X, activation='swish', norm='instance', b_train=b_train,
                                             scope='discriminator', use_patch=False)

    feature_loss = get_residual_loss(feature_real, feature_fake, type='l2')
    encoder_loss = get_residual_loss(X, fake_X, type='l2')
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

    alpha = 1.0
    total_loss = encoder_loss + alpha * feature_loss

    encoder_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(total_loss, var_list=[encoder_vars])

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            try:
                variables_to_restore = [v for v in tf.trainable_variables()
                                        if v.name.split('/')[0] == 'generator'
                                        or v.name.split('/')[0] == 'discriminator']
                saver = tf.train.Saver(variables_to_restore)
                print('Start New Training. Wait ...')
            except:
                print('Model load failed. No Pretrained GAN.')
                return

        trX = os.listdir(train_data)
        print('Number of Training Images: ' + str(len(trX)))
        num_augmentations = 1  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations

        for e in range(num_epoch):
            trX = shuffle(trX)
            training_batch = zip(range(0, len(trX), file_batch_size),
                                 range(file_batch_size, len(trX) + 1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                imgs = load_images(trX[start:end], base_dir=train_data, use_augmentation=False)
                imgs = np.expand_dims(imgs, axis=3)

                _, e_loss, decoded_images = sess.run([encoder_optimizer, total_loss, fake_X], feed_dict={X: imgs, b_train: True})

                print('epoch: ' + str(e) + ', loss: ' + str(e_loss))

                decoded_images = np.squeeze(decoded_images)
                cv2.imwrite('imgs/' + trX[start], decoded_images[3] * 255)
                # cv2.imwrite('imgs/org_' + trX[start], imgs[3] * 255)

                itr += 1

                if itr % 200 == 0:
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')

                if is_exit() is True:
                    return

            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')


def train_gan(model_path):
    print('Please wait. It takes several minutes. Do not quit!')

    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])
        Z = tf.placeholder(tf.float32, [batch_size, representation_dim])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Content discriminator
    fake_X = generator(Z, activation='swish', norm='instance', b_train=b_train, scope='generator', use_upsample=True)
    # Adversarial Discriminator
    feature_fake, logit_fake = discriminator(fake_X, activation='swish', norm='instance', b_train=b_train, scope='discriminator', use_patch=False)
    feature_real, logit_real = discriminator(X, activation='swish', norm='instance', b_train=b_train, scope='discriminator', use_patch=False)

    feature_loss = get_residual_loss(feature_real, feature_fake, type='l2')
    disc_loss, disc_loss_real, disc_loss_fake = get_discriminator_loss(logit_real, logit_fake, type='wgan')
    gen_loss = -tf.reduce_mean(logit_fake)

    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    # Alert: Clip range is critical to WGAN.
    disc_weight_clip = [p.assign(tf.clip_by_value(p, -0.05, 0.05)) for p in disc_vars]
    generator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

    alpha = 1.0

    # WGAN GP
    eps = tf.random_uniform([batch_size, input_height, input_width, num_channel], minval=0.0, maxval=1.0)
    gp_input = eps * X + (1.0 - eps) * fake_X
    _, gp_output = discriminator(gp_input, activation='swish', norm='instance', b_train=b_train, scope='discriminator', use_patch=False)
    gp_grad = tf.gradients(gp_output, [gp_input])[0]
    gp_grad_norm = tf.sqrt(tf.reduce_mean(gp_grad ** 2, axis=1), name="gp_grad_norm")
    gp_grad_pen = 10 * tf.reduce_mean((gp_grad_norm - 1) ** 2)

    feature_loss = alpha * feature_loss
    #disc_loss = disc_loss + gp_grad_pen

    disc_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_loss, var_list=disc_vars)
    gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_loss, var_list=generator_vars)

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Start New Training. Wait ...')

        trX = os.listdir(train_data)
        print('Number of Training Images: ' + str(len(trX)))
        num_augmentations = 1  # How many augmentations per 1 sample
        file_batch_size = batch_size // num_augmentations
        num_critic = 5

        for e in range(num_epoch):
            trX = shuffle(trX)
            training_batch = zip(range(0, len(trX), file_batch_size),  range(file_batch_size, len(trX)+1, file_batch_size))
            itr = 0

            for start, end in training_batch:
                imgs = load_images(trX[start:end], base_dir=train_data, use_augmentation=False)
                imgs = np.expand_dims(imgs, axis=3)

                noise = generate_sample_z(low=-1.0, high=1.0, num_samples=batch_size, sample_length=representation_dim,
                                          b_uniform=True)
                _, d_loss, _ = sess.run([disc_optimizer, disc_loss, disc_weight_clip], feed_dict={X: imgs, Z: noise, b_train: True})

                if itr % num_critic == 0:
                    noise = generate_sample_z(low=-1.0, high=1.0, num_samples=batch_size,
                                              sample_length=representation_dim,
                                              b_uniform=True)
                    _, g_loss = sess.run([gen_optimizer, gen_loss], feed_dict={Z: noise,  b_train: True})
                    # _, f_loss = sess.run([feature_optimizer, feature_loss], feed_dict={X: imgs, Z: noise, b_train: True})
                    decoded_images = sess.run([fake_X], feed_dict={Z: noise, b_train: True})
                    print('epoch: ' + str(e) + ', discriminator: ' + str(d_loss) +
                          ', generator: ' + str(g_loss))

                    decoded_images = np.squeeze(decoded_images)
                    cv2.imwrite('imgs/' + trX[start], (decoded_images[3] * 128.0) + 128.0)
                    #cv2.imwrite('imgs/org_' + trX[start], imgs[3] * 255)

                itr += 1

                if itr % 200 == 0:
                    try:
                        print('Saving model...')
                        saver.save(sess, model_path)
                        print('Saved.')
                    except:
                        print('Save failed')

                if is_exit() is True:
                    return

            try:
                print('Saving model...')
                saver.save(sess, model_path)
                print('Saved.')
            except:
                print('Save failed')


def test(model_path):
    with tf.device('/device:CPU:0'):
        X = tf.placeholder(tf.float32, [batch_size, input_height, input_width, num_channel])

    b_train = tf.placeholder(tf.bool)

    # Launch the graph in a session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    latent = encoder(X, activation='swish', norm='layer', b_train=b_train, scope='encoder')
    fake_X = generator(latent, activation='swish', norm='layer', b_train=b_train, scope='generator',
                       use_upsample=True)
    # Adversarial Discriminator
    feature_fake, logit_fake = discriminator(fake_X, activation='swish', norm='layer', b_train=b_train,
                                             scope='discriminator', use_patch=False)
    feature_real, logit_real = discriminator(X, activation='swish', norm='layer', b_train=b_train,
                                             scope='discriminator', use_patch=False)

    feature_loss = get_residual_loss(feature_real, feature_fake, type='l2')
    encoder_loss = get_residual_loss(X, fake_X, type='l1')
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')

    alpha = 1.0
    total_loss = encoder_loss + alpha * feature_loss

    # Launch the graph in a session
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        try:
            saver = tf.train.Saver()
            saver.restore(sess, model_path)
            print('Model Restored')
        except:
            print('Model Restore Failed')
            return

        trX = os.listdir(test_data)
        print('Number of Test Images: ' + str(len(trX)))
        file_batch_size = 1
        training_batch = zip(range(0, len(trX), file_batch_size),  range(file_batch_size, len(trX)+1, file_batch_size))

        for start, end in training_batch:
            imgs = load_images(trX[start:end], base_dir=test_data)
            imgs = np.expand_dims(imgs, axis=3)
            #print('Image batch Shape: ' + str(imgs.shape))
            aae_loss = sess.run([total_loss], feed_dict={X: imgs, b_train: False})

            print(str(trX[start:end]) + ' score: ' + str(aae_loss))


def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def is_exit(marker='.exit'):
    import os.path

    if os.path.isfile(marker) is True:
        os.remove(marker)
        return True

    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, help='train_gan/train_encoder/test', default='train_gan')
    parser.add_argument('--model_path', type=str, help='model check point file path', default='./model/m.ckpt')
    parser.add_argument('--train_data', type=str, help='training data directory', default='input')
    parser.add_argument('--test_data', type=str, help='test data directory', default='test')

    args = parser.parse_args()

    train_data = args.train_data
    test_data = args.test_data
    model_path = args.model_path

    dense_block_depth = 64

    # Bottle neck(depth narrow down) depth. See Residual Dense Block and Residual Block.
    bottleneck_depth = 32
    batch_size = 16
    representation_dim = 128

    img_width = 256
    img_height = 256
    input_width = 128
    input_height = 128
    num_channel = 1

    test_size = 100
    num_epoch = 30000

    if args.mode == 'train_gan':
        train_gan(model_path)
    elif args.mode == 'train_encoder':
        train_encoder(model_path)
    else:
        model_path = args.model_path
        test_data = args.test_data
        batch_size = 1
        test(model_path)
